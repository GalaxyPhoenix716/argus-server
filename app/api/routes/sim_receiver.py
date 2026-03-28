"""
ARGUS Simulation Receiver
WebSocket endpoint that accepts telemetry from argus_simulation
and broadcasts it to all connected dashboard clients.

Phase 2: Threat generation now flows through the AI pipeline:
  telemetry → AnomalyDetector → FeatureExtractor → ThreatClassifier → threat event
The simulation's mode string no longer directly drives threat events.
"""

import json
import logging
import time
from typing import Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import numpy as np

# Import the dashboard client connection manager from streaming
from app.api.routes.streaming import manager

# Import AI Engine components
from app.ai.AnomalyDetectionEngine.anomaly_detector import AnomalyDetector
from app.ai.AnomalyDetectionEngine.helpers.schemas import AnomalyResult
from app.ai.ThreatClassificationEngine.threat_classifier import ThreatClassifier
from app.ai.ThreatClassificationEngine.config.threat_config import get_config
from app.api.routes.blockchain import record_threat_log

logger = logging.getLogger(__name__)

router = APIRouter(tags=["simulation"])

# ── Channel list (must match simulation's 25-channel schema) ──────────────────
TELEMETRY_CHANNELS = [
    'position_lat', 'position_lon',
    'velocity_x', 'velocity_y', 'velocity_z',
    'altitude',
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'temperature', 'pressure', 'humidity',
    'battery_level', 'signal_strength',
    'gyro_x', 'gyro_y', 'gyro_z',
    'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
    'attitude_roll', 'attitude_pitch', 'attitude_yaw',
    'angular_velocity', 'timestamp',
]
CHANNEL_IDS = [f"CH_{i}" for i in range(len(TELEMETRY_CHANNELS))]
CHANNEL_MAP = dict(zip(TELEMETRY_CHANNELS, CHANNEL_IDS))

# Channels monitored for anomaly detection
MONITOR_CHANNELS = ['battery_level', 'temperature', 'signal_strength']

# Detection stability controls to reduce false positives.
ANOMALY_MIN_SEQUENCE_LEN = 8
ANOMALY_RECENT_TAIL = 12
ANOMALY_MIN_STREAK = 2
THREAT_EMIT_COOLDOWN_SEC = 20.0


# ── Statistical Fallback Detector (used when LSTM models unavailable) ──────────
class StatisticalAnomalyDetector:
    """
    Lightweight statistical anomaly detector for real-time telemetry.
    Uses rolling z-score for sudden changes AND linear trend slope for slow drifts.
    Works on any single scalar channel without requiring a trained model.
    """

    def __init__(
        self,
        channel_name: str,
        window: int = 50,
        k: float = 3.0,
        std_floor: float = 1e-6,
        min_slope_threshold: Optional[float] = None,
        max_slope_threshold: Optional[float] = None,
    ):
        self.channel_name = channel_name
        self.window = window
        self.k = k
        self.std_floor = max(std_floor, 1e-6)
        # Channel-specific slope thresholds (per frame).
        # If set, values crossing these indicate sustained drift anomalies.
        self.min_slope_threshold = min_slope_threshold
        self.max_slope_threshold = max_slope_threshold
        self._buffer: list = []
        self._threshold: float = 0.0
        self._mean: float = 0.0
        self._std: float = 1.0
        self._baseline_slope: float = 0.0   # learned normal slope
        self._slope_buffer: list = []       # rolling window of slopes

    def update(self, value: float) -> tuple[bool, float]:
        """
        Update detector with new value.
        Returns (is_anomaly, score).
        """
        self._buffer.append(value)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        if len(self._buffer) < 10:
            return False, 0.0

        self._mean = np.mean(self._buffer)
        self._std = max(np.std(self._buffer), self.std_floor)
        score = abs(value - self._mean) / self._std

        is_anomaly = score > self.k
        self._threshold = self.k * self._std

        return is_anomaly, score

    def _compute_slope(self, values: np.ndarray) -> float:
        """Compute linear trend slope (per-frame) using OLS."""
        if len(values) < 5:
            return 0.0
        t = np.arange(len(values))
        # Simple linear regression: slope = cov(t,y) / var(t)
        t_mean = np.mean(t)
        y_mean = np.mean(values)
        slope = np.sum((t - t_mean) * (values - y_mean)) / max(np.sum((t - t_mean) ** 2), 1e-9)
        return slope

    def detect_window(self, values: np.ndarray) -> list[tuple[int, int]]:
        """
        Scan a value series for anomalous windows using BOTH z-score and slope.
        Z-score is computed against the FIRST self.window values as baseline (pre-attack).
        Returns list of (start, end) index pairs.
        """
        if len(values) < 10:
            return []

        # Use first `window` values as the known-normal baseline for z-score
        baseline = values[:min(self.window, len(values) // 2)]
        if len(baseline) < 5:
            return []

        baseline_mean = np.mean(baseline)
        baseline_std = max(np.std(baseline), self.std_floor)

        # Z-score against PRE-ATTACK baseline (not including attack region)
        scores = np.abs(values - baseline_mean) / baseline_std
        z_above = scores > self.k

        # Trend slope on the latter portion of the window
        slope_window_size = min(30, len(values) // 2)
        if slope_window_size >= 5:
            recent_values = values[-slope_window_size:]
            recent_slope = self._compute_slope(recent_values)
        else:
            recent_slope = 0.0

        slope_anomaly = False
        if self.min_slope_threshold is not None and recent_slope < self.min_slope_threshold:
            slope_anomaly = True
        if self.max_slope_threshold is not None and recent_slope > self.max_slope_threshold:
            slope_anomaly = True

        # Combine: flag if either z-score OR slope is anomalous.
        # If slope anomaly is true, mark the recent section as anomalous.
        above = z_above.copy()
        if slope_anomaly:
            slope_start = max(0, len(above) - slope_window_size)
            above[slope_start:] = True

        # Build contiguous sequences
        sequences = []
        in_seq = False
        start = 0

        for i, v in enumerate(above):
            if v and not in_seq:
                start = i
                in_seq = True
            elif not v and in_seq:
                sequences.append((start, i))
                in_seq = False

        if in_seq:
            sequences.append((start, len(above)))

        return sequences

    def get_last_slope(self) -> float:
        """Return the most recently computed slope (for logging)."""
        if len(self._buffer) < 5:
            return 0.0
        return self._compute_slope(self._buffer[-30:])


# ── In-memory simulation state ────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.latest_telemetry: Optional[Dict] = None
        self.current_mode: str = "NORMAL"
        self.previous_mode: str = "NORMAL"
        self.sim_ws: Optional[WebSocket] = None
        self.sim_connected: bool = False
        self.threat_counter: int = 0
        self.start_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        # 250-timestep rolling buffer: (250, 25) numpy array
        self.telemetry_buffer: Optional[np.ndarray] = None
        self.frame_index: int = 0
        self._buffer_lock: asyncio.Lock = asyncio.Lock()
        self.channel_alert_state: Dict[str, Dict[str, float]] = {
            ch: {"streak": 0.0, "last_emit": 0.0} for ch in MONITOR_CHANNELS
        }

sim_state = SimState()

# ── AI Engine Initialization ─────────────────────────────────────────────────
logger.info("Initializing AI engines...")

# Per-channel LSTM anomaly detectors (loaded from model files)
_ad_detectors: Dict[str, AnomalyDetector] = {}
for ch in MONITOR_CHANNELS:
    try:
        detector = AnomalyDetector(CHANNEL_MAP[ch])
        detector.load_model()
        _ad_detectors[ch] = detector
        logger.info(f"LSTM AnomalyDetector loaded for channel: {ch}")
    except Exception as e:
        logger.warning(f"Could not load LSTM model for {ch}: {e} — using statistical fallback")

# Statistical fallback detectors (used when LSTM models unavailable)
_stat_detectors: Dict[str, StatisticalAnomalyDetector] = {}
_channel_stat_config: Dict[str, Dict[str, float]] = {
    # Battery has low natural jitter; clamp std to avoid huge z-scores in NORMAL.
    "battery_level": {
        "std_floor": 0.4,
        "min_slope_threshold": -0.18,
        "k": 4.2,
    },
    # Temperature drifts up under thermal anomalies.
    "temperature": {
        "std_floor": 0.35,
        "max_slope_threshold": 0.25,
        "k": 3.2,
    },
    # Signal has higher noise; use larger floor and stricter z-threshold.
    "signal_strength": {
        "std_floor": 1.5,
        "min_slope_threshold": -0.25,
        "k": 3.8,
    },
}
for ch in MONITOR_CHANNELS:
    if ch not in _ad_detectors:
        stat_cfg = _channel_stat_config.get(ch, {})
        _stat_detectors[ch] = StatisticalAnomalyDetector(
            ch,
            window=50,
            k=stat_cfg.get("k", 3.0),
            std_floor=stat_cfg.get("std_floor", 1e-6),
            min_slope_threshold=stat_cfg.get("min_slope_threshold"),
            max_slope_threshold=stat_cfg.get("max_slope_threshold"),
        )
        logger.info(f"Statistical fallback detector initialized for channel: {ch}")

# Threat classifier (Phase 2)
try:
    _threat_classifier = ThreatClassifier(config=get_config())
    logger.info("ThreatClassifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ThreatClassifier: {e}")
    _threat_classifier = None


# ── Attack metadata (for mapping AI classification → UI display) ─────────────
# NOTE: This maps AI-classified threat types to frontend display properties.
# It is NOT used to generate threats directly from simulation mode.
ATTACK_INFO: Dict[str, Dict[str, Any]] = {
    "GPS_SPOOFING":    {"type": "gps_spoofing",           "severity": "high",     "description": "GPS signal spoofing detected. Orbital position data is being falsified.",                          "affected_sensors": ["position_lat", "position_lon", "altitude"]},
    "SIGNAL_JAMMING":  {"type": "communication_loss",      "severity": "critical", "description": "RF jamming attack active. Downlink communication link severely degraded.",                            "affected_sensors": ["signal_strength"]},
    "THERMAL_ANOMALY":{"type": "thermal_anomaly",          "severity": "medium",   "description": "Thermal anomaly detected. Subsystem overheating beyond safe operating range.",                        "affected_sensors": ["temperature", "battery_level"]},
    "POWER_DRAIN":    {"type": "battery_degradation",      "severity": "high",     "description": "Rapid battery degradation detected. Power subsystem output compromised.",                        "affected_sensors": ["battery_level"]},
    "DDOS":           {"type": "multi_sensor_corruption",  "severity": "critical", "description": "DDoS attack saturating telemetry downlink. Data integrity compromised.",                        "affected_sensors": ["signal_strength", "timestamp"]},
    "COMMAND_INJECTION":{"type": "trajectory_manipulation", "severity": "critical","description": "Unauthorized command injection. ADCS attitude control systems compromised.",                      "affected_sensors": ["attitude_roll", "attitude_pitch", "attitude_yaw", "gyro_x", "gyro_y", "gyro_z"]},
    "SENSOR_FREEZE":  {"type": "sensor_stuck",            "severity": "high",     "description": "Sensor replay attack. All channels frozen — stale data being transmitted.",                      "affected_sensors": ["position_lat", "position_lon", "altitude", "temperature"]},
}


# ── Broadcast helpers ─────────────────────────────────────────────────────────

async def push(type_: str, payload: dict):
    """Broadcast a typed message to all dashboard WebSocket clients."""
    await manager.broadcast_to_all({
        "type": type_,
        "payload": payload,
    })


# ── Threat Classification Pipeline ────────────────────────────────────────────

def _buffer_to_feature_array(buffer: np.ndarray, ch_name: str) -> np.ndarray:
    """Extract a single channel's scalar series from the 25-ch buffer."""
    if ch_name not in CHANNEL_MAP:
        return np.array([])
    idx = TELEMETRY_CHANNELS.index(ch_name)
    return buffer[:, idx]


async def _run_lstm_pipeline(ch_name: str, values: np.ndarray,
                               detector: AnomalyDetector) -> Optional[AnomalyResult]:
    """
    Run LSTM-based anomaly detection on a scalar channel series.
    Returns AnomalyResult with detected sequences, or None if no anomaly.
    """
    try:
        l_s = detector.config.l_s
        # Normalize to [-1, 1] approximate range for LSTM
        normalized = values / 100.0 * 2.0 - 1.0

        # Pad if sequence shorter than l_s
        if len(normalized) < l_s:
            pad_val = normalized[0] if len(normalized) > 0 else 0.0
            normalized = np.concatenate([np.full(l_s - len(normalized), pad_val), normalized])

        # Take last l_s points for prediction
        normalized = normalized[-l_s:]

        # Build (T, 1) input — pad to at least 2*l_s so predictor can slide
        data = normalized.reshape(-1, 1)
        if len(data) < 2 * l_s:
            data = np.vstack([np.full((l_s, 1), normalized[0]), data])

        y_hat, y_true = detector.predict(data)

        errors_smoothed = detector.compute_errors(y_true, y_hat)
        threshold = detector.compute_threshold(errors_smoothed)
        sequences = detector.detect_anomalies(errors_smoothed, threshold)

        result = AnomalyResult(
            channel_id=ch_name,
            anomaly_sequences=sequences,
            errors_smoothed=errors_smoothed,
            threshold=threshold,
            y_hat=y_hat,
            y_true=y_true,
        )
        return result
    except Exception as e:
        logger.debug(f"LSTM pipeline failed for {ch_name}: {e}")
        return None


async def _run_statistical_pipeline(ch_name: str, values: np.ndarray,
                                     detector: StatisticalAnomalyDetector
                                     ) -> Optional[AnomalyResult]:
    """
    Run statistical anomaly detection on a scalar channel series.
    Returns AnomalyResult with detected sequences, or None if no anomaly.
    """
    try:
        # Score the entire window against a stable baseline.
        baseline = values[:min(detector.window, len(values) // 2)]
        if len(baseline) < 5:
            return None
        baseline_mean = np.mean(baseline)
        baseline_std = max(np.std(baseline), detector.std_floor)
        scores = np.abs(values - baseline_mean) / baseline_std
        sequences = detector.detect_window(values)

        # Build a synthetic AnomalyResult from statistical detection
        if not sequences:
            return None

        result = AnomalyResult(
            channel_id=ch_name,
            anomaly_sequences=sequences,
            errors_smoothed=scores,
            threshold=detector.k,
            y_hat=np.full_like(scores, 0.0),
            y_true=values,
        )
        return result
    except Exception as e:
        logger.debug(f"Statistical pipeline failed for {ch_name}: {e}")
        return None


async def run_ai_pipeline(buffer: np.ndarray):
    """
    Phase 2 AI Pipeline:
      1. Run AnomalyDetector (LSTM or statistical fallback) on monitored channels
      2. For each detected anomaly, run FeatureExtractor + ThreatClassifier
      3. Broadcast the AI-classified threat event

    Threats are generated exclusively by this pipeline — NOT from simulation mode.
    """
    if _threat_classifier is None:
        logger.warning("ThreatClassifier not available, skipping AI pipeline")
        return

    if buffer is None or len(buffer) < 30:
        return

    now = time.time()

    for ch_name in MONITOR_CHANNELS:
        result: Optional[AnomalyResult] = None
        state = sim_state.channel_alert_state.setdefault(
            ch_name, {"streak": 0.0, "last_emit": 0.0}
        )

        # Try LSTM first, fall back to statistical
        lstm_detector = _ad_detectors.get(ch_name)
        stat_detector = _stat_detectors.get(ch_name)

        values = _buffer_to_feature_array(buffer, ch_name)
        if len(values) == 0:
            continue

        if lstm_detector is not None:
            result = await _run_lstm_pipeline(ch_name, values, lstm_detector)
        elif stat_detector is not None:
            result = await _run_statistical_pipeline(ch_name, values, stat_detector)

        if result is None or not result.anomaly_sequences:
            state["streak"] = 0.0
            continue

        actionable_sequences = [
            (s, e)
            for (s, e) in result.anomaly_sequences
            if _is_actionable_sequence(max(0, s), min(len(buffer), e), len(buffer))
        ]
        if not actionable_sequences:
            state["streak"] = 0.0
            continue

        state["streak"] += 1.0
        if state["streak"] < ANOMALY_MIN_STREAK:
            continue

        if (now - state["last_emit"]) < THREAT_EMIT_COOLDOWN_SEC:
            continue

        logger.info(f"[AI Pipeline] {len(actionable_sequences)} actionable anomaly sequence(s) "
                    f"detected on channel: {ch_name}")

        # Classify the most recent actionable sequence only.
        seq_start, seq_end = actionable_sequences[-1]
        for seq_start, seq_end in [(seq_start, seq_end)]:
            # Guard array bounds
            seq_start = max(0, seq_start)
            seq_end = min(len(buffer), seq_end)
            if seq_end - seq_start < 3:
                continue

            # Extra hard gate for battery: only severe drops should emit.
            if ch_name == "battery_level":
                battery_window = values[seq_start:seq_end]
                if not _is_significant_battery_degradation(battery_window):
                    state["streak"] = 0.0
                    continue

            try:
                input_seq = buffer[seq_start:seq_end]

                # Build aligned y_true/y_hat/error_signal for FeatureExtractor
                buf_len = len(buffer)
                y_true_len = len(result.y_true)
                y_hat_len = len(result.y_hat)
                err_len = len(result.errors_smoothed)

                y_true_win = result.y_true[max(0, seq_start):seq_end] \
                    if seq_end <= y_true_len \
                    else np.concatenate([
                        result.y_true,
                        np.zeros(seq_end - y_true_len)
                    ])

                y_hat_win = result.y_hat[max(0, seq_start):seq_end] \
                    if seq_end <= y_hat_len \
                    else np.concatenate([
                        result.y_hat,
                        np.zeros(seq_end - y_hat_len)
                    ])

                err_win = result.errors_smoothed[max(0, seq_start):seq_end] \
                    if seq_end <= err_len \
                    else np.concatenate([
                        result.errors_smoothed,
                        np.zeros(seq_end - err_len)
                    ])

                # ── Phase 2: Classify via ThreatClassifier ──────────────────
                classification = _threat_classifier.classify_anomaly(
                    anomaly_result=result,
                    input_sequence=input_seq,
                    y_true=y_true_win,
                    y_hat=y_hat_win,
                    error_signal=err_win,
                    window_start=0,
                    window_end=len(input_seq),
                    temporal_context={"channel": ch_name, "buffer_len": buf_len}
                )

                logger.info(
                    f"[AI Pipeline] Classified: {classification.threat_class} "
                    f"(conf={classification.confidence:.2f}, "
                    f"type={classification.specific_threat_type}, "
                    f"method={classification.classification_method.value})"
                )

                # Only broadcast meaningful detections.
                # Keep "unknown" classes conservative to reduce noise.
                if classification.confidence < 0.35:
                    continue

                # Map to display properties
                threat_type = _normalize_threat_type(
                    ch_name, classification.specific_threat_type
                )
                severity = _map_confidence_to_severity(classification.confidence)

                sim_state.threat_counter += 1
                threat_id = f"threat-{sim_state.threat_counter}-{int(now)}"

                # ── Broadcast AI-classified threat ──────────────────────────────
                await push("threat", {
                    "id": threat_id,
                    "type": threat_type,
                    "severity": severity,
                    "confidence": round(classification.confidence, 2),
                    "risk_score": round(classification.risk_score, 2),
                    "affected_sensors": [ch_name],
                    "detected_at": now,
                    "description": (classification.reasoning or f"Anomaly detected on {ch_name}"),
                    "is_active": True,
                    "ai_classified": True,
                    "classification_method": classification.classification_method.value,
                })

                await push("event", {
                    "id": f"evt-{int(now * 1000)}",
                    "timestamp": now,
                    "type": "threat",
                    "message": f"[AI ENGINE] {classification.threat_class.upper()} - {threat_type}",
                    "details": (f"Channel: {ch_name} | "
                               f"Confidence: {classification.confidence:.2f} | "
                               f"Method: {classification.classification_method.value}"),
                })

                # Hash + store threat log in blockchain layer (or local fallback),
                # then stream verification status to frontend blockchain panel.
                blockchain_payload = {
                    "threat_id": threat_id,
                    "threat_type": threat_type,
                    "threat_class": classification.threat_class,
                    "confidence": round(classification.confidence, 2),
                    "risk_score": round(classification.risk_score, 2),
                    "channel": ch_name,
                    "detected_at": now,
                }
                blockchain_result = record_threat_log("threat", blockchain_payload)
                await push("blockchain", blockchain_result)

                state["last_emit"] = now

            except Exception as e:
                logger.error(f"[AI Pipeline] Classification error for {ch_name}: {e}", exc_info=True)


def _map_confidence_to_severity(confidence: float) -> str:
    """Map AI confidence score to severity level for frontend display."""
    if confidence >= 0.85:
        return "critical"
    elif confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    return "low"


def _is_actionable_sequence(seq_start: int, seq_end: int, buffer_len: int) -> bool:
    """
    Keep only ongoing, meaningful anomaly windows.
    This avoids replaying old windows and suppresses tiny blips.
    """
    seq_len = seq_end - seq_start
    is_recent = seq_end >= (buffer_len - ANOMALY_RECENT_TAIL)
    return seq_len >= ANOMALY_MIN_SEQUENCE_LEN and is_recent


def _is_significant_battery_degradation(window_values: np.ndarray) -> bool:
    """
    Battery alerts should require a clearly meaningful drop, not small drift.
    """
    if len(window_values) < ANOMALY_MIN_SEQUENCE_LEN:
        return False
    drop = float(window_values[0] - window_values[-1])
    t = np.arange(len(window_values))
    slope = float(np.polyfit(t, window_values, 1)[0]) if len(window_values) >= 5 else 0.0
    return drop >= 4.0 and slope <= -0.12


def _normalize_threat_type(channel_name: str, classified_type: Optional[str]) -> str:
    """
    Constrain noisy generic outputs (for example random sensor_injection)
    to channel-appropriate threat classes.
    """
    allowed = {
        "battery_level": {"battery_degradation", "sensor_drift", "sensor_stuck"},
        "temperature": {"thermal_anomaly", "sensor_drift", "sensor_stuck"},
        "signal_strength": {"communication_loss", "jamming", "multi_sensor_corruption"},
    }

    fallback = _fallback_threat_type_for_channel(channel_name)
    if not classified_type:
        return fallback
    if classified_type in allowed.get(channel_name, set()):
        return classified_type
    return fallback


def _fallback_threat_type_for_channel(channel_name: str) -> str:
    """Fallback threat type when classifier cannot assign a specific subtype."""
    if channel_name == "battery_level":
        return "battery_degradation"
    if channel_name == "temperature":
        return "thermal_anomaly"
    if channel_name == "signal_strength":
        return "communication_loss"
    return "sensor_compromise"


# ── Mode-change handler (info-only; threats come from AI pipeline) ─────────────

async def _on_mode_change(new_mode: str, old_mode: str):
    """
    Handle simulation mode transitions.
    NOTE: This does NOT generate threat events directly.
    Threat events now come exclusively from the AI pipeline (run_ai_pipeline).
    """
    now = time.time()

    if new_mode != "NORMAL" and new_mode in ATTACK_INFO:
        logger.info(
            f"[SIM MODE] Scenario changed to {new_mode}; waiting for AI-confirmed threat output"
        )
        await push("event", {
            "id": f"evt-{int(now * 1000)}",
            "timestamp": now,
            "type": "info",
            "message": "[SIM] Scenario changed; awaiting AI threat confirmation",
            "details": "No threat is emitted from simulator mode changes. Threats are AI-generated only.",
        })

    elif new_mode == "NORMAL" and old_mode != "NORMAL":
        await push("event", {
            "id": f"evt-{int(now * 1000)}",
            "timestamp": now,
            "type": "info",
            "message": "System returned to NOMINAL state",
            "details": "Scenario reset acknowledged; AI monitors for residual anomalies.",
        })
        logger.info("Attack mode cleared: %s -> NORMAL", old_mode)


# ── /ws/sim — WebSocket for simulation process ────────────────────────────────

@router.websocket("/ws/sim")
async def simulation_websocket(websocket: WebSocket):
    """
    The argus_simulation process connects here. It sends telemetry frames
    and the server broadcasts them to all dashboard clients.
    Also receives control commands forwarded from the dashboard.
    """
    await websocket.accept()
    sim_state.sim_ws = websocket
    sim_state.sim_connected = True
    sim_state.start_time = time.time()
    sim_state.telemetry_buffer = None
    sim_state.frame_index = 0
    sim_state.channel_alert_state = {
        ch: {"streak": 0.0, "last_emit": 0.0} for ch in MONITOR_CHANNELS
    }
    logger.info("Simulation process connected")

    await push("event", {
        "id": f"sim-connect-{int(time.time() * 1000)}",
        "timestamp": time.time(),
        "type": "info",
        "message": "Simulation process connected — telemetry stream live",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "telemetry":
                frame: dict = data.get("payload", {})
                mode: str = data.get("mode", "NORMAL")

                sim_state.latest_telemetry = frame
                sim_state.last_frame_time = time.time()

                # Build 25-channel feature vector from frame
                feature_vec = np.array([[frame.get(ch, 0.0) for ch in TELEMETRY_CHANNELS]])

                async with sim_state._buffer_lock:
                    if sim_state.telemetry_buffer is None:
                        sim_state.telemetry_buffer = feature_vec
                    else:
                        sim_state.telemetry_buffer = np.vstack([
                            sim_state.telemetry_buffer, feature_vec
                        ])
                        if sim_state.telemetry_buffer.shape[0] > 250:
                            sim_state.telemetry_buffer = sim_state.telemetry_buffer[-250:]

                # Broadcast raw telemetry to dashboard
                await push("telemetry", frame)

                # Trigger AI pipeline every 5 frames
                if sim_state.frame_index > 0 and sim_state.frame_index % 5 == 0:
                    async with sim_state._buffer_lock:
                        buf_copy = (sim_state.telemetry_buffer.copy()
                                    if sim_state.telemetry_buffer is not None else None)
                    if buf_copy is not None:
                        asyncio.create_task(run_ai_pipeline(buf_copy))

                sim_state.frame_index += 1

                # Mode change detection — informational only
                if mode != sim_state.current_mode:
                    old = sim_state.current_mode
                    sim_state.current_mode = mode
                    await _on_mode_change(mode, old)

    except WebSocketDisconnect:
        logger.info("Simulation process disconnected")
    except Exception as e:
        logger.error("Simulation WS error: %s", e, exc_info=True)
    finally:
        sim_state.sim_ws = None
        sim_state.sim_connected = False
        sim_state.current_mode = "NORMAL"
        sim_state.telemetry_buffer = None
        sim_state.frame_index = 0
        sim_state.channel_alert_state = {
            ch: {"streak": 0.0, "last_emit": 0.0} for ch in MONITOR_CHANNELS
        }

        await push("event", {
            "id": f"sim-disconnect-{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "type": "warning",
            "message": "Simulation process disconnected",
        })


# ── Control endpoint — dashboard → server → simulation ────────────────────────

class ControlRequest(BaseModel):
    action: str
    mode: Optional[str] = None
    speed: Optional[float] = None


@router.post("/api/v1/simulation/control")
async def control_simulation(request: ControlRequest):
    """Forward a control command to the simulation process via WebSocket."""
    if not sim_state.sim_connected or sim_state.sim_ws is None:
        return {"success": False, "error": "Simulation not connected"}

    try:
        await sim_state.sim_ws.send_text(json.dumps({
            "type": "control",
            "action": request.action,
            "mode": request.mode,
            "speed": request.speed,
        }))
        return {"success": True}
    except Exception as e:
        logger.error("Failed to send control command: %s", e)
        return {"success": False, "error": str(e)}


@router.get("/api/v1/simulation/status")
async def simulation_status():
    return {
        "connected": sim_state.sim_connected,
        "current_mode": sim_state.current_mode,
        "uptime_seconds": (time.time() - sim_state.start_time) if sim_state.start_time else 0,
        "last_frame_age_seconds": (time.time() - sim_state.last_frame_time) if sim_state.last_frame_time else None,
    }


@router.get("/api/v1/telemetry/latest")
async def latest_telemetry():
    return sim_state.latest_telemetry or {}


