"""
Feature Extractor for Threat Classification

Extracts engineered features from anomaly detection results for use in threat classification.
Supports consistency, residual, temporal, and correlation feature groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyFeatures:
    """Consistency-based features (GPS/velocity mismatch, sensor correlations)"""
    gps_velocity_mismatch: float
    position_velocity_correlation: float
    sensor_cross_validation_error: float
    expected_vs_actual_displacement: float
    acceleration_consistency: float
    sensor_agreement_score: float


@dataclass
class ResidualFeatures:
    """Residual-based features (LSTM prediction errors, reconstruction)"""
    mean_absolute_error: float
    max_error: float
    error_variance: float
    error_trend: float
    reconstruction_error: float
    prediction_confidence: float


@dataclass
class TemporalFeatures:
    """Temporal pattern features (rolling stats, trend, change points)"""
    rolling_mean_deviation: float
    rolling_std_deviation: float
    trend_slope: float
    trend_r_squared: float
    change_point_score: float
    volatility_index: float
    stability_score: float


@dataclass
class CorrelationFeatures:
    """Cross-sensor correlation features"""
    inter_sensor_correlation: float
    pca_reconstruction_error: float
    anomaly_isolation_score: float
    correlation_breakdown_score: float


@dataclass
class AllFeatures:
    """Complete feature set for threat classification"""
    consistency: ConsistencyFeatures
    residual: ResidualFeatures
    temporal: TemporalFeatures
    correlation: CorrelationFeatures

    def to_array(self) -> np.ndarray:
        """Convert all features to a flat numpy array for ML models"""
        features = []

        # Consistency features
        features.extend([
            self.consistency.gps_velocity_mismatch,
            self.consistency.position_velocity_correlation,
            self.consistency.sensor_cross_validation_error,
            self.consistency.expected_vs_actual_displacement,
            self.consistency.acceleration_consistency,
            self.consistency.sensor_agreement_score,
        ])

        # Residual features
        features.extend([
            self.residual.mean_absolute_error,
            self.residual.max_error,
            self.residual.error_variance,
            self.residual.error_trend,
            self.residual.reconstruction_error,
            self.residual.prediction_confidence,
        ])

        # Temporal features
        features.extend([
            self.temporal.rolling_mean_deviation,
            self.temporal.rolling_std_deviation,
            self.temporal.trend_slope,
            self.temporal.trend_r_squared,
            self.temporal.change_point_score,
            self.temporal.volatility_index,
            self.temporal.stability_score,
        ])

        # Correlation features
        features.extend([
            self.correlation.inter_sensor_correlation,
            self.correlation.pca_reconstruction_error,
            self.correlation.anomaly_isolation_score,
            self.correlation.correlation_breakdown_score,
        ])

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of all features in order"""
        return [
            'gps_velocity_mismatch',
            'position_velocity_correlation',
            'sensor_cross_validation_error',
            'expected_vs_actual_displacement',
            'acceleration_consistency',
            'sensor_agreement_score',
            'mean_absolute_error',
            'max_error',
            'error_variance',
            'error_trend',
            'reconstruction_error',
            'prediction_confidence',
            'rolling_mean_deviation',
            'rolling_std_deviation',
            'trend_slope',
            'trend_r_squared',
            'change_point_score',
            'volatility_index',
            'stability_score',
            'inter_sensor_correlation',
            'pca_reconstruction_error',
            'anomaly_isolation_score',
            'correlation_breakdown_score',
        ]


class FeatureExtractor:
    """
    Extracts engineered features from anomaly detection results.

    Takes the output from AnomalyDetector and extracts features for threat classification.
    """

    def __init__(self, n_features: int = 25):
        """
        Initialize feature extractor.

        Args:
            n_features: Number of features in the telemetry data
        """
        self.n_features = n_features
        self.pca = PCA(n_components=min(10, n_features))
        self.scaler = StandardScaler()
        self._fitted = False

        # Feature names mapping
        self.feature_name_map = {
            0: 'position_lat',
            1: 'position_lon',
            2: 'velocity_x',
            3: 'velocity_y',
            4: 'velocity_z',
            5: 'altitude',
            6: 'acceleration_x',
            7: 'acceleration_y',
            8: 'acceleration_z',
            9: 'temperature',
            10: 'pressure',
            11: 'humidity',
            12: 'battery_level',
            13: 'signal_strength',
            14: 'gyro_x',
            15: 'gyro_y',
            16: 'gyro_z',
            17: 'magnetometer_x',
            18: 'magnetometer_y',
            19: 'magnetometer_z',
            20: 'attitude_roll',
            21: 'attitude_pitch',
            22: 'attitude_yaw',
            23: 'angular_velocity',
            24: 'timestamp',
        }

    def extract_all_features(
        self,
        anomaly_result: Any,
        input_sequence: np.ndarray,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        error_signal: np.ndarray,
        window_start: int,
        window_end: int
    ) -> AllFeatures:
        """
        Extract all feature groups from anomaly detection results.

        Args:
            anomaly_result: Output from AnomalyDetector.run()
            input_sequence: Full input sequence (timesteps, n_features)
            y_true: True values (timesteps,)
            y_hat: Predicted values (timesteps,)
            error_signal: Smoothed error signal
            window_start: Start index of anomaly window
            window_end: End index of anomaly window

        Returns:
            AllFeatures object with extracted features
        """
        logger.debug(f"Extracting features for window [{window_start}, {window_end}]")

        # Extract each feature group
        consistency = self._extract_consistency_features(
            input_sequence, window_start, window_end
        )
        residual = self._extract_residual_features(
            y_true, y_hat, error_signal, window_start, window_end
        )
        temporal = self._extract_temporal_features(
            input_sequence, error_signal, window_start, window_end
        )
        correlation = self._extract_correlation_features(
            input_sequence, window_start, window_end
        )

        return AllFeatures(
            consistency=consistency,
            residual=residual,
            temporal=temporal,
            correlation=correlation
        )

    def _extract_consistency_features(
        self,
        input_sequence: np.ndarray,
        window_start: int,
        window_end: int
    ) -> ConsistencyFeatures:
        """
        Extract consistency-based features.

        Checks for GPS/velocity mismatches, sensor cross-validation, etc.
        """
        window_data = input_sequence[window_start:window_end]

        # GPS/velocity mismatch
        gps_velocity_mismatch = self._compute_gps_velocity_mismatch(window_data)

        # Position-velocity correlation
        position_velocity_correlation = self._compute_position_velocity_correlation(window_data)

        # Sensor cross-validation error
        sensor_cross_validation_error = self._compute_sensor_cross_validation(window_data)

        # Expected vs actual displacement
        expected_vs_actual_displacement = self._compute_displacement_consistency(window_data)

        # Acceleration consistency
        acceleration_consistency = self._compute_acceleration_consistency(window_data)

        # Sensor agreement score
        sensor_agreement_score = self._compute_sensor_agreement(window_data)

        return ConsistencyFeatures(
            gps_velocity_mismatch=gps_velocity_mismatch,
            position_velocity_correlation=position_velocity_correlation,
            sensor_cross_validation_error=sensor_cross_validation_error,
            expected_vs_actual_displacement=expected_vs_actual_displacement,
            acceleration_consistency=acceleration_consistency,
            sensor_agreement_score=sensor_agreement_score
        )

    def _compute_gps_velocity_mismatch(self, window_data: np.ndarray) -> float:
        """Compute mismatch between GPS position and velocity"""
        if window_data.shape[1] < 5:
            return 0.0

        # Extract position and velocity
        position_x = window_data[:, 0] if self.n_features > 0 else np.zeros(len(window_data))
        position_y = window_data[:, 1] if self.n_features > 1 else np.zeros(len(window_data))
        velocity_x = window_data[:, 2] if self.n_features > 2 else np.zeros(len(window_data))
        velocity_y = window_data[:, 3] if self.n_features > 3 else np.zeros(len(window_data))

        # Compute expected displacement from velocity
        dt = 1.0  # Assuming unit timesteps
        expected_dx = np.sum(velocity_x * dt)
        expected_dy = np.sum(velocity_y * dt)

        # Compute actual displacement
        actual_dx = position_x[-1] - position_x[0]
        actual_dy = position_y[-1] - position_y[0]

        # Compute mismatch
        expected_displacement = np.sqrt(expected_dx**2 + expected_dy**2)
        actual_displacement = np.sqrt(actual_dx**2 + actual_dy**2)

        if expected_displacement < 1e-10:
            return 0.0

        mismatch = abs(actual_displacement - expected_displacement) / (expected_displacement + 1e-10)
        return float(mismatch)

    def _compute_position_velocity_correlation(self, window_data: np.ndarray) -> float:
        """Compute correlation between position and velocity"""
        if window_data.shape[1] < 4:
            return 0.0

        # Position and velocity arrays
        position = window_data[:, :2] if window_data.shape[1] >= 2 else np.zeros((len(window_data), 2))
        velocity = window_data[:, 2:4] if window_data.shape[1] >= 4 else np.zeros((len(window_data), 2))

        # Compute correlation
        pos_flat = position.flatten()
        vel_flat = velocity.flatten()

        correlation = np.corrcoef(pos_flat, vel_flat)[0, 1]
        return float(abs(correlation)) if not np.isnan(correlation) else 0.0

    def _compute_sensor_cross_validation(self, window_data: np.ndarray) -> float:
        """Compute cross-validation error between redundant sensors"""
        if window_data.shape[1] < 2:
            return 0.0

        # Use first two similar sensors (e.g., two temperature sensors)
        sensor1 = window_data[:, 0]
        sensor2 = window_data[:, 1] if window_data.shape[1] > 1 else window_data[:, 0]

        # Compute difference
        diff = np.abs(sensor1 - sensor2)
        error = np.mean(diff) / (np.std(sensor1) + np.std(sensor2) + 1e-10)

        return float(error)

    def _compute_displacement_consistency(self, window_data: np.ndarray) -> float:
        """Compute consistency between expected and actual displacement"""
        if window_data.shape[1] < 2:
            return 0.0

        # Simple implementation: compare first and last position
        start_pos = window_data[0, :2] if window_data.shape[1] >= 2 else window_data[0, :1]
        end_pos = window_data[-1, :2] if window_data.shape[1] >= 2 else window_data[-1, :1]

        displacement = np.linalg.norm(end_pos - start_pos)

        # For this implementation, return the displacement magnitude
        return float(displacement)

    def _compute_acceleration_consistency(self, window_data: np.ndarray) -> float:
        """Compute consistency of acceleration measurements"""
        if window_data.shape[1] < 7:
            return 0.0

        # Acceleration measurements
        accel_x = window_data[:, 6] if window_data.shape[1] > 6 else np.zeros(len(window_data))
        accel_y = window_data[:, 7] if window_data.shape[1] > 7 else np.zeros(len(window_data))
        accel_z = window_data[:, 8] if window_data.shape[1] > 8 else np.zeros(len(window_data))

        # Compute acceleration magnitude
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Check for consistency (low variance indicates consistent acceleration)
        consistency = 1.0 / (1.0 + np.std(accel_mag))

        return float(consistency)

    def _compute_sensor_agreement(self, window_data: np.ndarray) -> float:
        """Compute how well sensors agree with each other"""
        if window_data.shape[1] < 2:
            return 1.0

        # Compute pairwise correlations
        correlations = []
        for i in range(min(window_data.shape[1], 5)):  # Limit to 5 sensors
            for j in range(i+1, min(window_data.shape[1], 5)):
                corr = np.corrcoef(window_data[:, i], window_data[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return 1.0

        return float(np.mean(correlations))

    def _extract_residual_features(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        error_signal: np.ndarray,
        window_start: int,
        window_end: int
    ) -> ResidualFeatures:
        """
        Extract residual-based features.

        Uses LSTM prediction errors and reconstruction quality.
        """
        # Error statistics
        mean_absolute_error = float(np.mean(error_signal[window_start:window_end]))
        max_error = float(np.max(error_signal[window_start:window_end]))
        error_variance = float(np.var(error_signal[window_start:window_end]))

        # Error trend
        error_window = error_signal[window_start:window_end]
        if len(error_window) > 1:
            trend = np.polyfit(range(len(error_window)), error_window, 1)[0]
        else:
            trend = 0.0
        error_trend = float(trend)

        # Reconstruction error (simplified)
        reconstruction_error = float(mean_absolute_error / (np.mean(y_true) + 1e-10))

        # Prediction confidence (inverse of error)
        prediction_confidence = float(1.0 / (1.0 + mean_absolute_error))

        return ResidualFeatures(
            mean_absolute_error=mean_absolute_error,
            max_error=max_error,
            error_variance=error_variance,
            error_trend=error_trend,
            reconstruction_error=reconstruction_error,
            prediction_confidence=prediction_confidence
        )

    def _extract_temporal_features(
        self,
        input_sequence: np.ndarray,
        error_signal: np.ndarray,
        window_start: int,
        window_end: int
    ) -> TemporalFeatures:
        """
        Extract temporal pattern features.

        Analyzes rolling statistics, trends, and change points.
        """
        window_data = input_sequence[window_start:window_end]
        error_window = error_signal[window_start:window_end]

        # Rolling mean deviation
        if len(window_data) > 10:
            rolling_mean = pd.Series(window_data[:, 0]).rolling(window=min(10, len(window_data)//2)).mean()
            mean_deviation = np.mean(np.abs(window_data[:, 0] - rolling_mean.dropna()))
        else:
            mean_deviation = 0.0
        rolling_mean_deviation = float(mean_deviation)

        # Rolling standard deviation
        if len(window_data) > 10:
            rolling_std = pd.Series(window_data[:, 0]).rolling(window=min(10, len(window_data)//2)).std()
            std_deviation = np.mean(rolling_std.dropna())
        else:
            std_deviation = float(np.std(window_data[:, 0]))
        rolling_std_deviation = float(std_deviation)

        # Trend slope
        if len(window_data) > 1:
            trend_slope = float(np.polyfit(range(len(window_data)), window_data[:, 0], 1)[0])
        else:
            trend_slope = 0.0

        # Trend R-squared
        if len(window_data) > 2:
            y = window_data[:, 0]
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))
        else:
            r_squared = 0.0
        trend_r_squared = float(r_squared)

        # Change point score
        change_point_score = float(self._compute_change_point_score(error_window))

        # Volatility index
        volatility_index = float(np.std(error_window))

        # Stability score (inverse of volatility)
        stability_score = float(1.0 / (1.0 + volatility_index))

        return TemporalFeatures(
            rolling_mean_deviation=rolling_mean_deviation,
            rolling_std_deviation=rolling_std_deviation,
            trend_slope=trend_slope,
            trend_r_squared=trend_r_squared,
            change_point_score=change_point_score,
            volatility_index=volatility_index,
            stability_score=stability_score
        )

    def _compute_change_point_score(self, error_window: np.ndarray) -> float:
        """Detect change points in error signal"""
        if len(error_window) < 3:
            return 0.0

        # Compute differences
        diffs = np.abs(np.diff(error_window))

        # Score based on magnitude of differences
        score = np.mean(diffs) / (np.std(error_window) + 1e-10)

        return float(score)

    def _extract_correlation_features(
        self,
        input_sequence: np.ndarray,
        window_start: int,
        window_end: int
    ) -> CorrelationFeatures:
        """
        Extract correlation-based features.

        Analyzes inter-sensor correlations and PCA reconstruction.
        """
        window_data = input_sequence[window_start:window_end]

        # Inter-sensor correlation
        if window_data.shape[1] >= 2:
            # Compute mean absolute correlation (excluding diagonal)
            corr_matrix = np.corrcoef(window_data.T)
            mask = np.ones_like(corr_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            correlations = corr_matrix[mask]
            inter_sensor_correlation = float(np.mean(np.abs(correlations)))
        else:
            inter_sensor_correlation = 0.0

        # PCA reconstruction error
        if not self._fitted and len(window_data) > 10:
            # Fit PCA on initial window
            self.pca.fit(window_data)
            self.scaler.fit(window_data)
            self._fitted = True

        if self._fitted:
            # Transform and inverse transform
            scaled_data = self.scaler.transform(window_data)
            pca_data = self.pca.transform(scaled_data)
            reconstructed = self.pca.inverse_transform(pca_data)
            reconstructed = self.scaler.inverse_transform(reconstructed)

            # Compute reconstruction error
            reconstruction_error = np.mean((window_data - reconstructed) ** 2)
        else:
            reconstruction_error = 0.0

        pca_reconstruction_error = float(reconstruction_error)

        # Anomaly isolation score
        anomaly_isolation_score = float(self._compute_anomaly_isolation(window_data))

        # Correlation breakdown score
        correlation_breakdown_score = float(self._compute_correlation_breakdown(window_data))

        return CorrelationFeatures(
            inter_sensor_correlation=inter_sensor_correlation,
            pca_reconstruction_error=pca_reconstruction_error,
            anomaly_isolation_score=anomaly_isolation_score,
            correlation_breakdown_score=correlation_breakdown_score
        )

    def _compute_anomaly_isolation(self, window_data: np.ndarray) -> float:
        """Compute how isolated the anomaly is"""
        if window_data.shape[1] < 2:
            return 1.0

        # Variance per feature
        variances = np.var(window_data, axis=0)

        # Isolation score: inverse of average variance
        avg_variance = np.mean(variances)
        isolation_score = 1.0 / (1.0 + avg_variance)

        return float(isolation_score)

    def _compute_correlation_breakdown(self, window_data: np.ndarray) -> float:
        """Detect breakdown in normal correlations"""
        if window_data.shape[1] < 2:
            return 0.0

        # Compute correlation matrix
        corr_matrix = np.corrcoef(window_data.T)

        # Look for unusual correlation patterns
        # For example, very high or very low correlations
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        correlations = corr_matrix[mask]

        # Score based on number of extreme correlations
        extreme_count = np.sum(np.abs(correlations) > 0.9) + np.sum(np.abs(correlations) < 0.1)
        breakdown_score = extreme_count / len(correlations) if len(correlations) > 0 else 0.0

        return float(breakdown_score)

    def batch_extract_features(
        self,
        anomaly_results: Dict[str, Any],
        input_data: Dict[str, np.ndarray]
    ) -> Dict[str, AllFeatures]:
        """
        Extract features for multiple channels in batch.

        Args:
            anomaly_results: Dict mapping channel_id to AnomalyResult
            input_data: Dict mapping channel_id to input sequence

        Returns:
            Dict mapping channel_id to AllFeatures
        """
        features_dict = {}

        for channel_id, result in anomaly_results.items():
            if channel_id not in input_data:
                logger.warning(f"No input data for channel {channel_id}")
                continue

            input_seq = input_data[channel_id]

            # Extract features for each anomaly window
            for window_start, window_end in result.anomaly_sequences:
                # Get the corresponding y_true, y_hat, and error_signal
                # Note: This would need to be adapted based on your AnomalyDetector output
                y_true = input_seq[window_start:window_end, 0] if input_seq.shape[1] > 0 else np.zeros(window_end - window_start)
                y_hat = y_true  # Simplified - use actual predictions in real implementation
                error_signal = result.errors_smoothed

                features = self.extract_all_features(
                    result, input_seq, y_true, y_hat, error_signal,
                    window_start, window_end
                )

                features_dict[f"{channel_id}_{window_start}"] = features

        return features_dict