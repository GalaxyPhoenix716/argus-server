"""
Threat Classification Orchestrator

Main orchestrator for threat classification that combines rule-based heuristics
and ML-based classification for robust threat detection.
"""

from enum import Enum
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging

from .feature_extractor import FeatureExtractor, AllFeatures
from .rule_based_heuristics import RuleBasedHeuristics, RuleBasedResult, ThreatType
from .threat_model import ThreatClassificationModel
from .config.threat_config import ThreatClassificationConfig, get_config

logger = logging.getLogger(__name__)


class ClassificationMethod(Enum):
    """Method used for classification"""
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"
    ML = "ml"
    UNKNOWN = "unknown"


@dataclass
class ThreatClassification:
    """
    Result of threat classification.

    Contains class prediction, confidence, risk score, and supporting information.
    """
    # Main classification
    threat_class: str  # "attack", "failure", "unknown"
    confidence: float  # [0, 1] - overall confidence
    risk_score: float  # [0, 1] - risk level

    # Classification details
    classification_method: ClassificationMethod
    ml_prediction: Optional[str] = None  # ML model prediction
    rule_based_prediction: Optional[str] = None  # Rule-based prediction

    # Supporting information
    top_features: Optional[List[Dict]] = None  # Top contributing features
    reasoning: Optional[str] = None  # Human-readable reasoning
    specific_threat_type: Optional[str] = None  # AttackType or FailureType value

    # Metadata
    processing_time_ms: Optional[float] = None
    rule_based_confidence: Optional[float] = None
    ml_confidence: Optional[float] = None


class ThreatClassifier:
    """
    Main threat classification orchestrator.

    Combines rule-based heuristics and ML-based classification for robust
    threat detection in aerospace/defense telemetry.
    """

    def __init__(
        self,
        config: Optional[ThreatClassificationConfig] = None,
        model: Optional[ThreatClassificationModel] = None
    ):
        """
        Initialize threat classifier.

        Args:
            config: Configuration (uses default if not provided)
            model: Pre-trained model (trains new one if None)
        """
        self.config = config or get_config()
        self.model = model

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.rule_based = RuleBasedHeuristics(self.config.rules)

        # State
        self.fitted = model is not None and model.fitted

        # Performance tracking
        self.processing_times = {
            'feature_extraction': [],
            'rule_based': [],
            'ml': [],
            'ensemble': []
        }

        logger.info("Threat classifier initialized")

    def classify_anomaly(
        self,
        anomaly_result: Any,
        input_sequence: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        y_hat: Optional[np.ndarray] = None,
        error_signal: Optional[np.ndarray] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
        temporal_context: Optional[Dict] = None
    ) -> ThreatClassification:
        """
        Classify a detected anomaly.

        Args:
            anomaly_result: Output from AnomalyDetector.run()
            input_sequence: Full input sequence (timesteps, n_features)
            y_true: True values (optional)
            y_hat: Predicted values (optional)
            error_signal: Smoothed error signal (optional)
            window_start: Start index of anomaly window
            window_end: End index of anomaly window
            temporal_context: Optional temporal analysis results

        Returns:
            ThreatClassification result
        """
        import time
        start_time = time.time()

        try:
            # Validate inputs
            if window_start is None or window_end is None:
                if hasattr(anomaly_result, 'anomaly_sequences') and anomaly_result.anomaly_sequences:
                    window_start, window_end = anomaly_result.anomaly_sequences[0]
                else:
                    # Default: use entire sequence
                    window_start = 0
                    window_end = len(input_sequence)

            # Extract features
            feature_extraction_start = time.time()
            features = self.feature_extractor.extract_all_features(
                anomaly_result,
                input_sequence,
                y_true or np.array([]),
                y_hat or np.array([]),
                error_signal or np.array([]),
                window_start,
                window_end
            )
            self.processing_times['feature_extraction'].append(
                (time.time() - feature_extraction_start) * 1000
            )

            # Rule-based classification
            rule_based_start = time.time()
            rule_result = self.rule_based.classify(features, temporal_context)
            self.processing_times['rule_based'].append(
                (time.time() - rule_based_start) * 1000
            )

            # ML-based classification (if model is available)
            ml_result = None
            ml_prediction = None
            ml_confidence = None

            if self.model is not None and self.model.fitted:
                ml_start = time.time()

                # Convert features to array
                features_array = features.to_array().reshape(1, -1)

                # Get prediction and confidence
                ml_pred, ml_proba, ml_conf = self.model.predict_with_confidence(
                    features_array,
                    confidence_threshold=self.config.classification.unknown_threshold
                )

                ml_prediction = self._convert_label_to_class(ml_pred[0])
                ml_confidence = float(ml_conf[0])

                self.processing_times['ml'].append(
                    (time.time() - ml_start) * 1000
                )

            # Ensemble decision
            ensemble_start = time.time()
            ensemble_result = self._make_ensemble_decision(
                rule_result, ml_prediction, ml_confidence
            )
            self.processing_times['ensemble'].append(
                (time.time() - ensemble_start) * 1000
            )

            # Get top features
            top_features = self._get_top_features(features, features_array if self.model else None)

            # Generate reasoning
            reasoning = self._generate_reasoning(
                ensemble_result, rule_result, ml_prediction, features
            )

            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000

            # Create result
            result = ThreatClassification(
                threat_class=ensemble_result['class'],
                confidence=ensemble_result['confidence'],
                risk_score=ensemble_result['risk_score'],
                classification_method=ensemble_result['method'],
                ml_prediction=ml_prediction,
                rule_based_prediction=rule_result.threat_type.value,
                top_features=top_features,
                reasoning=reasoning,
                specific_threat_type=rule_result.specific_type,
                processing_time_ms=total_time,
                rule_based_confidence=rule_result.confidence,
                ml_confidence=ml_confidence
            )

            logger.debug(
                f"Classified anomaly as {result.threat_class} "
                f"(confidence: {result.confidence:.2f}, "
                f"time: {total_time:.2f}ms)"
            )

            return result

        except Exception as e:
            logger.error(f"Error classifying anomaly: {e}", exc_info=True)

            # Return unknown on error
            return ThreatClassification(
                threat_class="unknown",
                confidence=0.0,
                risk_score=0.0,
                classification_method=ClassificationMethod.UNKNOWN,
                reasoning=f"Classification failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _make_ensemble_decision(
        self,
        rule_result: RuleBasedResult,
        ml_prediction: Optional[str],
        ml_confidence: Optional[float]
    ) -> Dict[str, Any]:
        """
        Make ensemble decision by combining rule-based and ML predictions.

        Args:
            rule_result: Rule-based classification result
            ml_prediction: ML model prediction
            ml_confidence: ML model confidence

        Returns:
            Ensemble decision dictionary
        """
        # If ML model is not available, use rule-based only
        if ml_prediction is None:
            return {
                'class': rule_result.threat_type.value,
                'confidence': rule_result.confidence,
                'risk_score': rule_result.risk_score,
                'method': ClassificationMethod.RULE_BASED
            }

        # Ensemble weights
        ml_weight = self.config.classification.ensemble_weights['ml']
        rule_weight = self.config.classification.ensemble_weights['rule_based']

        # Convert predictions to numeric for combination
        class_scores = {
            'attack': 0.0,
            'failure': 0.0,
            'unknown': 0.0
        }

        # Add rule-based score
        if rule_result.threat_type == ThreatType.ATTACK:
            class_scores['attack'] += rule_weight * rule_result.confidence
        elif rule_result.threat_type == ThreatType.FAILURE:
            class_scores['failure'] += rule_weight * rule_result.confidence
        else:
            class_scores['unknown'] += rule_weight * rule_result.confidence

        # Add ML score
        if ml_prediction in class_scores:
            class_scores[ml_prediction] += ml_weight * ml_confidence

        # Select class with highest score
        best_class = max(class_scores, key=class_scores.get)
        best_score = class_scores[best_class]

        # Calculate confidence
        if best_score > 0:
            confidence = min(best_score, 1.0)
        else:
            confidence = 0.0

        # Fallback to unknown if confidence too low
        if confidence < self.config.classification.unknown_threshold:
            return {
                'class': 'unknown',
                'confidence': confidence,
                'risk_score': 0.0,
                'method': ClassificationMethod.ENSEMBLE
            }

        # Calculate risk score
        if best_class == 'attack':
            risk_score = confidence * 1.2  # Higher risk for attacks
        elif best_class == 'failure':
            risk_score = confidence * 0.8  # Moderate risk for failures
        else:
            risk_score = confidence * 0.1  # Low risk for unknown

        risk_score = min(risk_score, 1.0)

        # Determine method
        method = ClassificationMethod.ENSEMBLE

        # If one method significantly outperforms, use that
        if ml_confidence is not None:
            if ml_confidence > 0.8 and rule_result.confidence < 0.5:
                method = ClassificationMethod.ML
            elif rule_result.confidence > 0.8 and ml_confidence < 0.5:
                method = ClassificationMethod.RULE_BASED

        return {
            'class': best_class,
            'confidence': confidence,
            'risk_score': risk_score,
            'method': method
        }

    def _convert_label_to_class(self, label: int) -> str:
        """Convert numeric label to class string"""
        # Assuming label encoding: 0=attack, 1=failure, 2=unknown
        # Adjust based on your training data
        class_map = {0: 'attack', 1: 'failure', 2: 'unknown'}
        return class_map.get(label, 'unknown')

    def _get_top_features(self, features: AllFeatures, features_array: Optional[np.ndarray]) -> List[Dict]:
        """
        Get top contributing features.

        Args:
            features: AllFeatures object
            features_array: Flattened features array

        Returns:
            List of top features with importance scores
        """
        top_features = []

        try:
            # Get feature names
            feature_names = features.get_feature_names()
            feature_values = features.to_array()

            # If model is available, use feature importances
            if self.model is not None and self.model.fitted:
                importances = self.model.feature_importances_

                # Create list of (name, importance, value) tuples
                feature_info = list(zip(feature_names, importances, feature_values))

                # Sort by importance
                feature_info.sort(key=lambda x: x[1], reverse=True)

                # Take top 5
                for name, importance, value in feature_info[:5]:
                    top_features.append({
                        'name': name,
                        'importance': float(importance),
                        'value': float(value)
                    })
            else:
                # Use rule-based importance scores
                # Convert features to importance scores (simplified)
                consistency_scores = {
                    'gps_velocity_mismatch': features.consistency.gps_velocity_mismatch,
                    'sensor_cross_validation_error': features.consistency.sensor_cross_validation_error,
                    'position_velocity_correlation': features.consistency.position_velocity_correlation,
                }

                residual_scores = {
                    'mean_absolute_error': features.residual.mean_absolute_error,
                    'max_error': features.residual.max_error,
                    'prediction_confidence': features.residual.prediction_confidence,
                }

                temporal_scores = {
                    'change_point_score': features.temporal.change_point_score,
                    'volatility_index': features.temporal.volatility_index,
                    'stability_score': features.temporal.stability_score,
                }

                correlation_scores = {
                    'inter_sensor_correlation': features.correlation.inter_sensor_correlation,
                    'anomaly_isolation_score': features.correlation.anomaly_isolation_score,
                }

                # Combine all scores
                all_scores = {}
                all_scores.update(consistency_scores)
                all_scores.update(residual_scores)
                all_scores.update(temporal_scores)
                all_scores.update(correlation_scores)

                # Sort by score
                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

                # Take top 5
                for name, score in sorted_scores[:5]:
                    top_features.append({
                        'name': name,
                        'importance': float(score),
                        'value': float(score)
                    })

        except Exception as e:
            logger.warning(f"Error getting top features: {e}")

        return top_features

    def _generate_reasoning(
        self,
        ensemble_result: Dict,
        rule_result: RuleBasedResult,
        ml_prediction: Optional[str],
        features: AllFeatures
    ) -> str:
        """
        Generate human-readable reasoning for classification.

        Args:
            ensemble_result: Ensemble decision
            rule_result: Rule-based result
            ml_prediction: ML prediction
            features: Extracted features

        Returns:
            Natural language explanation
        """
        parts = []

        # Header
        class_name = ensemble_result['class'].upper()
        parts.append(f"THREAT CLASSIFICATION: {class_name}")
        parts.append(f"Confidence: {ensemble_result['confidence']:.2f}")
        parts.append(f"Risk Score: {ensemble_result['risk_score']:.2f}")

        # Method used
        method = ensemble_result['method'].value
        parts.append(f"\nClassification Method: {method.upper()}")

        # Individual predictions
        parts.append(f"\nRule-based: {rule_result.threat_type.value} (confidence: {rule_result.confidence:.2f})")
        if ml_prediction is not None:
            parts.append(f"ML Model: {ml_prediction} (confidence: {rule_result.confidence:.2f})")

        # Reasoning from rule-based
        if rule_result.reasoning:
            parts.append(f"\n{rule_result.reasoning}")

        # Key features
        parts.append("\nKey Features:")
        parts.append(f"  • GPS/velocity mismatch: {features.consistency.gps_velocity_mismatch:.3f}")
        parts.append(f"  • Sensor variance: {features.consistency.sensor_cross_validation_error:.3f}")
        parts.append(f"  • Error magnitude: {features.residual.mean_absolute_error:.3f}")
        parts.append(f"  • Change point score: {features.temporal.change_point_score:.3f}")
        parts.append(f"  • Anomaly isolation: {features.correlation.anomaly_isolation_score:.3f}")

        # Recommendations
        if ensemble_result['class'] == 'attack':
            parts.append("\nRecommended Actions:")
            parts.append("  1. Verify signal authenticity")
            parts.append("  2. Check for external interference")
            parts.append("  3. Consider switching to backup systems")
            parts.append("  4. Alert security team immediately")
        elif ensemble_result['class'] == 'failure':
            parts.append("\nRecommended Actions:")
            parts.append("  1. Schedule maintenance")
            parts.append("  2. Check sensor calibration")
            parts.append("  3. Review system logs")
            parts.append("  4. Plan component replacement")

        return "\n".join(parts)

    def batch_classify(
        self,
        anomaly_results: Dict[str, Any],
        input_data: Dict[str, np.ndarray]
    ) -> Dict[str, ThreatClassification]:
        """
        Classify anomalies for multiple channels in batch.

        Args:
            anomaly_results: Dict mapping channel_id to AnomalyResult
            input_data: Dict mapping channel_id to input sequence

        Returns:
            Dict mapping channel_id to ThreatClassification
        """
        results = {}

        for channel_id, result in anomaly_results.items():
            try:
                if channel_id not in input_data:
                    logger.warning(f"No input data for channel {channel_id}")
                    continue

                input_seq = input_data[channel_id]

                # Get anomaly window (simplified - use first anomaly)
                if result.anomaly_sequences:
                    window_start, window_end = result.anomaly_sequences[0]

                    classification = self.classify_anomaly(
                        anomaly_result=result,
                        input_sequence=input_seq,
                        window_start=window_start,
                        window_end=window_end
                    )

                    results[channel_id] = classification

            except Exception as e:
                logger.error(f"Error classifying channel {channel_id}: {e}")

                # Add error result
                results[channel_id] = ThreatClassification(
                    threat_class="unknown",
                    confidence=0.0,
                    risk_score=0.0,
                    classification_method=ClassificationMethod.UNKNOWN,
                    reasoning=f"Classification error: {str(e)}"
                )

        logger.info(f"Classified {len(results)} channels")
        return results

    def get_processing_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get processing time statistics.

        Returns:
            Dictionary with processing time statistics per component
        """
        stats = {}
        for component, times in self.processing_times.items():
            if times:
                stats[component] = {
                    'mean_ms': float(np.mean(times)),
                    'std_ms': float(np.std(times)),
                    'min_ms': float(np.min(times)),
                    'max_ms': float(np.max(times)),
                    'count': len(times)
                }

        return stats

    def set_model(self, model: ThreatClassificationModel):
        """Set the ML model for classification"""
        self.model = model
        self.fitted = model is not None and model.fitted

    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Get classifier information.

        Returns:
            Dictionary with classifier information
        """
        info = {
            'fitted': self.fitted,
            'has_model': self.model is not None,
            'model_fitted': self.model.fitted if self.model else False,
            'config': {
                'unknown_threshold': self.config.classification.unknown_threshold,
                'attack_threshold': self.config.classification.attack_confidence_threshold,
                'failure_threshold': self.config.classification.failure_confidence_threshold,
            }
        }

        if self.model is not None and self.model.fitted:
            info['model_info'] = self.model.get_model_info()

        return info