"""
Temporal Pattern Analyzer

Detects temporal patterns in anomaly signals for explainability.
Classifies anomalies as spike, drift, persistent, or intermittent.
"""

from enum import Enum
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TemporalPattern(str, Enum):
    """Temporal pattern types"""
    SPIKE = "spike"
    DRIFT = "drift"
    PERSISTENT = "persistent"
    INTERMITTENT = "intermittent"


@dataclass
class TemporalAnalysis:
    """Results of temporal pattern analysis"""
    pattern_type: TemporalPattern
    onset_time: int
    duration: int
    intensity: float
    trend_slope: float
    stability_score: float
    volatility_index: float
    change_point_score: float
    monotonic_score: float
    burst_count: Optional[int] = None
    burst_intensity: Optional[float] = None


class TemporalPatternDetector:
    """
    Detects temporal patterns in anomaly sequences.

    Analyzes error signals to classify temporal patterns:
    - Spike: Sudden, brief anomaly
    - Drift: Gradual deviation
    - Persistent: Sustained anomaly
    - Intermittent: Multiple bursts
    """

    def __init__(self, min_duration: int = 3, sensitivity: float = 0.5):
        """
        Initialize temporal pattern detector.

        Args:
            min_duration: Minimum duration for pattern detection
            sensitivity: Detection sensitivity (0-1)
        """
        self.min_duration = min_duration
        self.sensitivity = sensitivity

    def analyze_pattern(
        self,
        errors: np.ndarray,
        anomaly_window: Tuple[int, int],
        threshold: float
    ) -> TemporalAnalysis:
        """
        Analyze temporal pattern in anomaly window.

        Args:
            errors: Full error signal
            anomaly_window: (start, end) indices of anomaly
            threshold: Detection threshold

        Returns:
            TemporalAnalysis with pattern classification
        """
        start, end = anomaly_window
        window_errors = errors[start:end]

        if len(window_errors) < self.min_duration:
            # Too short, treat as spike
            return TemporalAnalysis(
                pattern_type=TemporalPattern.SPIKE,
                onset_time=start,
                duration=len(window_errors),
                intensity=float(np.max(window_errors)),
                trend_slope=0.0,
                stability_score=1.0,
                volatility_index=float(np.std(window_errors)),
                change_point_score=0.0,
                monotonic_score=0.0
            )

        # Compute pattern characteristics
        characteristics = self._compute_characteristics(window_errors, threshold)

        # Classify pattern
        pattern_type = self._classify_pattern(window_errors, characteristics)

        return TemporalAnalysis(
            pattern_type=pattern_type,
            onset_time=start,
            duration=len(window_errors),
            intensity=float(np.max(window_errors)),
            trend_slope=characteristics['trend_slope'],
            stability_score=characteristics['stability_score'],
            volatility_index=characteristics['volatility_index'],
            change_point_score=characteristics['change_point_score'],
            monotonic_score=characteristics['monotonic_score'],
            burst_count=characteristics.get('burst_count'),
            burst_intensity=characteristics.get('burst_intensity')
        )

    def _compute_characteristics(
        self,
        window_errors: np.ndarray,
        threshold: float
    ) -> dict:
        """
        Compute temporal characteristics.

        Args:
            window_errors: Error signal in anomaly window
            threshold: Detection threshold

        Returns:
            Dictionary of characteristics
        """
        characteristics = {}

        # Basic statistics
        characteristics['max_error'] = float(np.max(window_errors))
        characteristics['mean_error'] = float(np.mean(window_errors))
        characteristics['std_error'] = float(np.std(window_errors))
        characteristics['volatility_index'] = characteristics['std_error']

        # Trend slope
        if len(window_errors) > 1:
            characteristics['trend_slope'] = float(np.polyfit(range(len(window_errors)), window_errors, 1)[0])
        else:
            characteristics['trend_slope'] = 0.0

        # Change point score
        characteristics['change_point_score'] = self._compute_change_point_score(window_errors)

        # Stability score
        if characteristics['mean_error'] > 1e-10:
            characteristics['stability_score'] = 1.0 - (characteristics['volatility_index'] / characteristics['mean_error'])
        else:
            characteristics['stability_score'] = 1.0

        # Monotonic score
        characteristics['monotonic_score'] = self._compute_monotonic_score(window_errors)

        # Burst detection (for intermittent patterns)
        bursts = self._detect_bursts(window_errors, threshold)
        if bursts:
            characteristics['burst_count'] = len(bursts)
            characteristics['burst_intensity'] = float(np.mean([burst[2] for burst in bursts]))
        else:
            characteristics['burst_count'] = 0
            characteristics['burst_intensity'] = 0.0

        return characteristics

    def _classify_pattern(
        self,
        window_errors: np.ndarray,
        characteristics: dict
    ) -> TemporalPattern:
        """
        Classify temporal pattern based on characteristics.

        Args:
            window_errors: Error signal
            characteristics: Pre-computed characteristics

        Returns:
            TemporalPattern classification
        """
        n_points = len(window_errors)
        intensity = characteristics['max_error']
        trend_slope = abs(characteristics['trend_slope'])
        stability = characteristics['stability_score']
        monotonic_score = characteristics['monotonic_score']
        burst_count = characteristics.get('burst_count', 0)

        # Decision tree for pattern classification
        # Based on characteristics and thresholds

        # Spike: High intensity, short duration, sudden onset
        if n_points < 10 and intensity > 2.0 * self.sensitivity:
            return TemporalPattern.SPIKE

        # Intermittent: Multiple bursts
        if burst_count >= 2:
            return TemporalPattern.INTERMITTENT

        # Drift: Monotonic trend, gradual change
        if monotonic_score > 0.7 and trend_slope > 0.3 * self.sensitivity:
            return TemporalPattern.DRIFT

        # Persistent: Sustained deviation, stable intensity
        if stability > 0.5 and burst_count <= 1:
            return TemporalPattern.PERSISTENT

        # Default classification based on dominant characteristic
        if trend_slope > 0.5:
            return TemporalPattern.DRIFT
        elif stability < 0.3:
            return TemporalPattern.INTERMITTENT
        else:
            return TemporalPattern.PERSISTENT

    def _compute_change_point_score(self, errors: np.ndarray) -> float:
        """
        Compute change point score.

        Detects sudden changes in error signal.

        Args:
            errors: Error signal

        Returns:
            Change point score
        """
        if len(errors) < 3:
            return 0.0

        # Compute differences
        diffs = np.abs(np.diff(errors))

        # Score based on magnitude of differences
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # High change point score if differences are large
        if std_diff > 1e-10:
            score = mean_diff / std_diff
        else:
            score = 0.0

        return float(score)

    def _compute_monotonic_score(self, errors: np.ndarray) -> float:
        """
        Compute monotonic score.

        Measures how monotonic the error signal is.

        Args:
            errors: Error signal

        Returns:
            Monotonic score [0, 1]
        """
        if len(errors) < 3:
            return 0.0

        # Count direction changes
        direction_changes = 0
        prev_direction = None

        for i in range(1, len(errors)):
            curr_direction = np.sign(errors[i] - errors[i-1])

            if curr_direction != 0:
                if prev_direction is not None and curr_direction != prev_direction:
                    direction_changes += 1
                prev_direction = curr_direction

        # Monotonic score is inverse of direction changes
        max_changes = len(errors) - 1
        if max_changes == 0:
            return 1.0

        monotonic_score = 1.0 - (direction_changes / max_changes)
        return float(monotonic_score)

    def _detect_bursts(
        self,
        errors: np.ndarray,
        threshold: float
    ) -> list:
        """
        Detect burst patterns in error signal.

        Args:
            errors: Error signal
            threshold: Detection threshold

        Returns:
            List of bursts (start, end, intensity)
        """
        if len(errors) == 0:
            return []

        # Find points above threshold
        above_threshold = errors > threshold

        if not np.any(above_threshold):
            return []

        # Find bursts
        bursts = []
        in_burst = False
        burst_start = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_burst:
                in_burst = True
                burst_start = i
            elif not is_above and in_burst:
                in_burst = False
                burst_end = i - 1
                intensity = float(np.max(errors[burst_start:burst_end + 1]))
                bursts.append((burst_start, burst_end, intensity))

        # Handle case where signal ends in a burst
        if in_burst:
            burst_end = len(errors) - 1
            intensity = float(np.max(errors[burst_start:burst_end + 1]))
            bursts.append((burst_start, burst_end, intensity))

        return bursts

    def batch_analyze(
        self,
        errors: np.ndarray,
        anomaly_windows: list,
        thresholds: list
    ) -> list:
        """
        Analyze multiple anomaly windows in batch.

        Args:
            errors: Full error signal
            anomaly_windows: List of (start, end) tuples
            thresholds: List of detection thresholds

        Returns:
            List of TemporalAnalysis results
        """
        results = []

        for i, (window, threshold) in enumerate(zip(anomaly_windows, thresholds)):
            try:
                analysis = self.analyze_pattern(errors, window, threshold)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing window {i}: {e}")
                # Return default analysis on error
                results.append(TemporalAnalysis(
                    pattern_type=TemporalPattern.PERSISTENT,
                    onset_time=window[0],
                    duration=window[1] - window[0],
                    intensity=0.0,
                    trend_slope=0.0,
                    stability_score=0.0,
                    volatility_index=0.0,
                    change_point_score=0.0,
                    monotonic_score=0.0
                ))

        return results

    def get_pattern_summary(self, analysis: TemporalAnalysis) -> dict:
        """
        Get human-readable summary of pattern analysis.

        Args:
            analysis: TemporalAnalysis result

        Returns:
            Dictionary with pattern summary
        """
        summary = {
            'pattern': analysis.pattern_type.value,
            'duration_timesteps': analysis.duration,
            'intensity': f"{analysis.intensity:.3f}",
            'trend': 'increasing' if analysis.trend_slope > 0 else 'decreasing' if analysis.trend_slope < 0 else 'stable',
            'stability': f"{analysis.stability_score:.3f}",
            'volatility': f"{analysis.volatility_index:.3f}"
        }

        # Add pattern-specific information
        if analysis.pattern_type == TemporalPattern.SPIKE:
            summary['description'] = 'Sudden, brief anomaly with high intensity'
        elif analysis.pattern_type == TemporalPattern.DRIFT:
            summary['description'] = 'Gradual deviation with clear trend'
        elif analysis.pattern_type == TemporalPattern.PERSISTENT:
            summary['description'] = 'Sustained anomaly with consistent intensity'
        elif analysis.pattern_type == TemporalPattern.INTERMITTENT:
            summary['description'] = f'Multiple bursts ({analysis.burst_count} detected)'
            summary['burst_intensity'] = f"{analysis.burst_intensity:.3f}"

        return summary