"""
ARGUS Anomaly Detection Engine
================================
Implements the TELEMANOM algorithm (Hundman et al., KDD 2018) adapted for the
ARGUS mission security system.

This file is the **thin orchestrator** that composes the helpers/ sub-modules
into the public AnomalyDetector and BatchAnomalyDetector classes.

Pipeline (per channel):
    1. data_loader    → load raw .npy telemetry
    2. model_loader   → load pre-trained LSTM (.h5 / .onnx)
    3. predictor      → sliding-window batch inference → y_hat, y_true
    4. signal_processing → compute errors + EWMA smoothing + dynamic threshold
    5. sequence_utils → extract, buffer, merge anomaly windows
    6. evaluator      → TP / FP / FN vs labeled_anomalies.csv

For full implementations see:
    helpers/config.py           — hyperparameters & paths
    helpers/schemas.py          — AnomalyResult dataclass
    helpers/backend.py          — TF / ONNX backend detection
    helpers/data_loader.py      — load_data(), load_cached_yhat()
    helpers/model_loader.py     — load_model()
    helpers/predictor.py        — predict()
    helpers/signal_processing.py — compute_errors(), compute_threshold()
    helpers/sequence_utils.py   — detect_anomalies() + primitives
    helpers/evaluator.py        — evaluate()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from helpers import (
    AnomalyResult,
    ChannelConfig,
    _DATA_DIR,
    _DEFAULT_MODEL_DIR,
    _LABELS_CSV,
    compute_errors,
    compute_threshold,
    detect_anomalies,
    evaluate,
    load_cached_yhat,
    load_data,
    load_model,
    predict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-channel detector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """
    Per-channel anomaly detector wrapping the full TELEMANOM pipeline.

    Parameters
    ----------
    channel_id : str
        e.g. "P-1", "A-7", "M-2"
    model_dir : Path | str | None
        Directory with <channel_id>.h5 (Keras) or <channel_id>.onnx files.
        Defaults to the bundled 2018 cached-run models directory.
    data_dir : Path | str | None
        Root data directory containing train/ and test/ sub-folders.
    config : ChannelConfig | None
        Hyperparameter overrides; uses paper defaults if None.
    """

    def __init__(
        self,
        channel_id: str,
        model_dir: Optional[Path | str] = None,
        data_dir: Optional[Path | str] = None,
        config: Optional[ChannelConfig] = None,
    ):
        self.channel_id = channel_id.upper()
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.config = config or ChannelConfig()

        self._model = None
        self.y_true: Optional[np.ndarray] = None
        self.y_hat: Optional[np.ndarray] = None
        self.errors: Optional[np.ndarray] = None
        self.errors_smoothed: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1 — Data
    # ------------------------------------------------------------------
    def load_data(self, split: str = "test") -> np.ndarray:
        """Load the .npy telemetry file. Returns shape (timesteps, features)."""
        return load_data(self.channel_id, split=split, data_dir=self.data_dir)

    def load_cached_yhat(self, run_dir: Optional[Path | str] = None) -> np.ndarray:
        """
        Load pre-computed y_hat from the 2018 cached run (no TF/GPU required).
        Also populates self.errors_smoothed if the smoothed_errors file exists.
        """
        y_hat, errors_smoothed = load_cached_yhat(
            self.channel_id,
            run_dir=Path(run_dir) if run_dir else None,
        )
        self.y_hat = y_hat
        self.errors_smoothed = errors_smoothed
        return y_hat

    # ------------------------------------------------------------------
    # Step 2 — Model
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the LSTM model (Keras .h5 or ONNX) for this channel."""
        self._model = load_model(self.channel_id, model_dir=self.model_dir)
        return self._model

    # ------------------------------------------------------------------
    # Step 3 — Prediction
    # ------------------------------------------------------------------
    def predict(self, data: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window LSTM inference.

        Returns (y_hat, y_true), each shape (T - l_s,).
        """
        if data is None:
            data = self.load_data("test")
        if self._model is None:
            self.load_model()

        y_hat, y_true = predict(
            self._model, data, self.config, channel_id=self.channel_id
        )
        self.y_hat = y_hat
        self.y_true = y_true
        return y_hat, y_true

    # ------------------------------------------------------------------
    # Step 4a — Errors & smoothing
    # ------------------------------------------------------------------
    def compute_errors(
        self,
        y_true: Optional[np.ndarray] = None,
        y_hat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute absolute errors and EWMA smooth. Returns errors_smoothed."""
        y_true = y_true if y_true is not None else self.y_true
        y_hat = y_hat if y_hat is not None else self.y_hat
        if y_true is None or y_hat is None:
            raise ValueError("Call predict() or load_cached_yhat() first.")

        self.errors, self.errors_smoothed = compute_errors(
            y_true, y_hat, self.config, channel_id=self.channel_id
        )
        return self.errors_smoothed

    # ------------------------------------------------------------------
    # Step 4b — Dynamic threshold
    # ------------------------------------------------------------------
    def compute_threshold(
        self, errors_smoothed: Optional[np.ndarray] = None
    ) -> float:
        """Non-parametric dynamic threshold selection."""
        e_s = errors_smoothed if errors_smoothed is not None else self.errors_smoothed
        if e_s is None:
            raise ValueError("Call compute_errors() first.")
        return compute_threshold(e_s, self.config, channel_id=self.channel_id)

    # ------------------------------------------------------------------
    # Step 5 — Anomaly window extraction
    # ------------------------------------------------------------------
    def detect_anomalies(
        self,
        errors_smoothed: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> list[tuple[int, int]]:
        """
        Detect anomaly windows from smoothed errors.
        Returns (start, end) pairs aligned to the original time axis.
        """
        e_s = errors_smoothed if errors_smoothed is not None else self.errors_smoothed
        if e_s is None:
            raise ValueError("Call compute_errors() first.")
        if threshold is None:
            threshold = self.compute_threshold(e_s)
        return detect_anomalies(
            e_s, threshold, self.config, channel_id=self.channel_id
        )

    # ------------------------------------------------------------------
    # Step 6 — Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        result: AnomalyResult,
        labels_csv: Optional[Path | str] = None,
    ) -> AnomalyResult:
        """Compute TP/FP/FN against labeled_anomalies.csv and update result."""
        return evaluate(
            result,
            self.channel_id,
            labels_csv=Path(labels_csv) if labels_csv else None,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, split: str = "test") -> AnomalyResult:
        """
        Execute the complete pipeline for this channel in one call.

        Returns an AnomalyResult with all intermediate artefacts.
        """
        data = self.load_data(split)
        y_hat, y_true = self.predict(data)
        e_s = self.compute_errors(y_true, y_hat)
        threshold = self.compute_threshold(e_s)
        sequences = self.detect_anomalies(e_s, threshold)

        return AnomalyResult(
            channel_id=self.channel_id,
            anomaly_sequences=sequences,
            errors_smoothed=e_s,
            threshold=threshold,
            y_hat=y_hat,
            y_true=y_true,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
class BatchAnomalyDetector:
    """
    Run the anomaly detector across multiple (or all) telemetry channels.

    Parameters
    ----------
    channel_ids : list[str] | None
        Channels to process. If None, all channels in labeled_anomalies.csv
        are discovered automatically.
    use_cached : bool
        Skip live inference and use pre-computed y_hat / smoothed_errors from
        the 2018 cached run (much faster; no GPU required).
    """

    def __init__(
        self,
        channel_ids: Optional[list[str]] = None,
        model_dir: Optional[Path | str] = None,
        data_dir: Optional[Path | str] = None,
        config: Optional[ChannelConfig] = None,
        use_cached: bool = False,
    ):
        if channel_ids is None:
            df = pd.read_csv(_LABELS_CSV)
            channel_ids = df["chan_id"].unique().tolist()

        self.channel_ids = channel_ids
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.config = config
        self.use_cached = use_cached
        self.results: dict[str, AnomalyResult] = {}

    def run_all(self, evaluate_results: bool = True) -> dict[str, AnomalyResult]:
        """Process all configured channels sequentially."""
        for ch_id in self.channel_ids:
            try:
                detector = AnomalyDetector(
                    ch_id,
                    model_dir=self.model_dir,
                    data_dir=self.data_dir,
                    config=self.config,
                )

                if self.use_cached:
                    result = self._run_cached(detector, ch_id)
                else:
                    result = detector.run()

                if evaluate_results:
                    result = detector.evaluate(result)

                self.results[ch_id] = result
                logger.info("Completed channel %s.", ch_id)

            except Exception as exc:
                logger.error("Error on channel %s: %s", ch_id, exc)

        self._log_totals()
        return self.results

    def _run_cached(self, detector: AnomalyDetector, ch_id: str) -> AnomalyResult:
        """Build an AnomalyResult from cached y_hat and smoothed errors."""
        y_hat = detector.load_cached_yhat()
        y_true_full = detector.load_data("test")[:, 0]
        cfg = detector.config
        y_true = y_true_full[cfg.l_s: cfg.l_s + len(y_hat)]
        detector.y_true = y_true

        # Prefer cached smoothed errors; recompute if missing
        e_s = detector.errors_smoothed
        if e_s is None:
            e_s = detector.compute_errors(y_true, y_hat)

        threshold = detector.compute_threshold(e_s)
        sequences = detector.detect_anomalies(e_s, threshold)

        return AnomalyResult(
            channel_id=ch_id,
            anomaly_sequences=sequences,
            errors_smoothed=e_s,
            threshold=threshold,
            y_hat=y_hat,
            y_true=y_true,
        )

    def _log_totals(self):
        tp = sum(r.true_positives for r in self.results.values())
        fp = sum(r.false_positives for r in self.results.values())
        fn = sum(r.false_negatives for r in self.results.values())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        logger.info(
            "=== BATCH TOTALS === TP:%d  FP:%d  FN:%d  P:%.4f  R:%.4f",
            tp, fp, fn, precision, recall,
        )

    def summary_df(self) -> pd.DataFrame:
        """Return a per-channel results DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results.values()])
