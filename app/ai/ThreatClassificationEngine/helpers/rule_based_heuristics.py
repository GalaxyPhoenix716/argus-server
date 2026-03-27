"""
Rule-Based Heuristics for Threat Classification

Domain-specific rules for classifying threats based on feature patterns.
Designed for aerospace/defense telemetry anomaly classification.
"""

from enum import Enum
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from .feature_extractor import AllFeatures
from .config.threat_config import RuleBasedConfig, get_config

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Threat types for classification"""
    ATTACK = "attack"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class AttackType(Enum):
    """Specific attack types"""
    GPS_SPOOFING = "gps_spoofing"
    SENSOR_INJECTION = "sensor_injection"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    MULTI_SENSOR_CORRUPTION = "multi_sensor_corruption"
    JAMMING = "jamming"


class FailureType(Enum):
    """Specific failure types"""
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_DEAD = "sensor_dead"
    COMMUNICATION_LOSS = "communication_loss"
    SYSTEM_DEGRADATION = "system_degradation"
    CALIBRATION_DRIFT = "calibration_drift"


@dataclass
class RuleBasedResult:
    """Result from rule-based classification"""
    threat_type: ThreatType
    specific_type: Optional[str]  # AttackType or FailureType value
    confidence: float
    reasoning: str
    triggered_rules: List[str]
    risk_score: float


class RuleBasedHeuristics:
    """
    Rule-based threat classification system.

    Uses domain knowledge and heuristics to classify anomalies as attacks, failures,
    or unknown. Designed for aerospace/defense applications with specific sensor
    patterns and attack vectors.
    """

    def __init__(self, config: Optional[RuleBasedConfig] = None):
        """
        Initialize rule-based heuristics.

        Args:
            config: Rule-based configuration (uses default if not provided)
        """
        self.config = config or get_config().rules
        self.threat_signatures = self._initialize_threat_signatures()

    def classify(
        self,
        features: AllFeatures,
        temporal_context: Optional[Dict] = None
    ) -> RuleBasedResult:
        """
        Classify threat using rule-based heuristics.

        Args:
            features: Extracted features from FeatureExtractor
            temporal_context: Optional temporal analysis results

        Returns:
            RuleBasedResult with classification and reasoning
        """
        # Try each threat category
        attack_result = self._detect_attack(features)
        if attack_result.confidence >= self.config.rule_confidence_base:
            return attack_result

        failure_result = self._detect_failure(features)
        if failure_result.confidence >= self.config.rule_confidence_base:
            return failure_result

        # If no clear pattern detected, return unknown
        return RuleBasedResult(
            threat_type=ThreatType.UNKNOWN,
            specific_type=None,
            confidence=self.config.rule_confidence_decay,
            reasoning="No clear pattern matched by rule-based heuristics",
            triggered_rules=[],
            risk_score=0.1
        )

    def _detect_attack(self, features: AllFeatures) -> RuleBasedResult:
        """
        Detect attack patterns.

        Checks for patterns typical of adversarial attacks:
        - GPS spoofing: position/velocity inconsistencies
        - Sensor injection: artificial signal patterns
        - Multi-sensor corruption: correlated anomalies across sensors
        """
        triggered_rules = []

        # Rule 1: GPS Spoofing Detection
        gps_spoofing_score = self._check_gps_spoofing(features)
        if gps_spoofing_score > 0:
            triggered_rules.append(f"GPS_SPOOFING(score={gps_spoofing_score:.2f})")

        # Rule 2: Sensor Injection Detection
        injection_score = self._check_sensor_injection(features)
        if injection_score > 0:
            triggered_rules.append(f"SENSOR_INJECTION(score={injection_score:.2f})")

        # Rule 3: Temporal Manipulation Detection
        manipulation_score = self._check_temporal_manipulation(features)
        if manipulation_score > 0:
            triggered_rules.append(f"TEMPORAL_MANIPULATION(score={manipulation_score:.2f})")

        # Rule 4: Multi-Sensor Corruption
        corruption_score = self._check_multi_sensor_corruption(features)
        if corruption_score > 0:
            triggered_rules.append(f"MULTI_SENSOR_CORRUPTION(score={corruption_score:.2f})")

        # Rule 5: Jamming Detection
        jamming_score = self._check_jamming(features)
        if jamming_score > 0:
            triggered_rules.append(f"JAMMING(score={jamming_score:.2f})")

        # Combine scores
        if triggered_rules:
            # Calculate confidence based on strongest signal
            max_score = max(
                gps_spoofing_score, injection_score, manipulation_score,
                corruption_score, jamming_score
            )

            # Boost confidence if multiple rules triggered
            confidence = min(
                self.config.rule_confidence_base + (0.1 * len(triggered_rules)),
                0.95
            ) * min(max_score, 1.0)

            # Generate reasoning
            specific_type = self._get_attack_type_from_scores(
                gps_spoofing_score, injection_score, manipulation_score,
                corruption_score, jamming_score
            )

            reasoning = self._generate_attack_reasoning(
                features, specific_type, triggered_rules
            )

            # Calculate risk score (high for attacks)
            risk_score = min(confidence * 1.2, 1.0)

            return RuleBasedResult(
                threat_type=ThreatType.ATTACK,
                specific_type=specific_type,
                confidence=confidence,
                reasoning=reasoning,
                triggered_rules=triggered_rules,
                risk_score=risk_score
            )

        return RuleBasedResult(
            threat_type=ThreatType.UNKNOWN,
            specific_type=None,
            confidence=0.0,
            reasoning="No attack patterns detected",
            triggered_rules=[],
            risk_score=0.0
        )

    def _detect_failure(self, features: AllFeatures) -> RuleBasedResult:
        """
        Detect failure patterns.

        Checks for patterns typical of system failures:
        - Sensor drift: gradual deviation over time
        - Sensor stuck: no variation in readings
        - Sensor dead: no signal or zero values
        - Communication loss: loss of signal
        """
        triggered_rules = []

        # Rule 1: Sensor Drift Detection
        drift_score = self._check_sensor_drift(features)
        if drift_score > 0:
            triggered_rules.append(f"SENSOR_DRIFT(score={drift_score:.2f})")

        # Rule 2: Sensor Stuck Detection
        stuck_score = self._check_sensor_stuck(features)
        if stuck_score > 0:
            triggered_rules.append(f"SENSOR_STUCK(score={stuck_score:.2f})")

        # Rule 3: Sensor Dead Detection
        dead_score = self._check_sensor_dead(features)
        if dead_score > 0:
            triggered_rules.append(f"SENSOR_DEAD(score={dead_score:.2f})")

        # Rule 4: Communication Loss Detection
        comm_loss_score = self._check_communication_loss(features)
        if comm_loss_score > 0:
            triggered_rules.append(f"COMMUNICATION_LOSS(score={comm_loss_score:.2f})")

        # Rule 5: System Degradation Detection
        degradation_score = self._check_system_degradation(features)
        if degradation_score > 0:
            triggered_rules.append(f"SYSTEM_DEGRADATION(score={degradation_score:.2f})")

        # Combine scores
        if triggered_rules:
            max_score = max(
                drift_score, stuck_score, dead_score,
                comm_loss_score, degradation_score
            )

            confidence = min(
                self.config.rule_confidence_base + (0.1 * len(triggered_rules)),
                0.9
            ) * min(max_score, 1.0)

            specific_type = self._get_failure_type_from_scores(
                drift_score, stuck_score, dead_score,
                comm_loss_score, degradation_score
            )

            reasoning = self._generate_failure_reasoning(
                features, specific_type, triggered_rules
            )

            # Calculate risk score (moderate for failures)
            risk_score = confidence * 0.8

            return RuleBasedResult(
                threat_type=ThreatType.FAILURE,
                specific_type=specific_type,
                confidence=confidence,
                reasoning=reasoning,
                triggered_rules=triggered_rules,
                risk_score=risk_score
            )

        return RuleBasedResult(
            threat_type=ThreatType.UNKNOWN,
            specific_type=None,
            confidence=0.0,
            reasoning="No failure patterns detected",
            triggered_rules=[],
            risk_score=0.0
        )

    def _check_gps_spoofing(self, features: AllFeatures) -> float:
        """
        Check for GPS spoofing indicators.

        GPS spoofing typically shows:
        - High GPS/velocity mismatch
        - Low position-velocity correlation
        - Inconsistent acceleration
        """
        score = 0.0

        # Check GPS/velocity mismatch
        if features.consistency.gps_velocity_mismatch > self.config.gps_spoofing_thresholds['gps_velocity_mismatch']:
            score += 0.3

        # Check position-velocity correlation
        if features.consistency.position_velocity_correlation < self.config.gps_spoofing_thresholds['position_velocity_correlation']:
            score += 0.3

        # Check acceleration consistency
        if features.consistency.acceleration_consistency < 0.5:
            score += 0.2

        # Check for sudden change (high change point score)
        if features.temporal.change_point_score > self.config.gps_spoofing_thresholds['sudden_change']:
            score += 0.2

        # Boost if multiple sensors affected
        if features.correlation.anomaly_isolation_score < 0.5:
            score += 0.1

        return min(score, 1.0)

    def _check_sensor_injection(self, features: AllFeatures) -> float:
        """
        Check for sensor injection attacks.

        Sensor injection shows:
        - Artificial signal patterns
        - Low reconstruction error
        - Pattern regularity
        """
        score = 0.0

        # Check if anomaly is too "clean" (low reconstruction error might indicate injection)
        if 0.01 < features.residual.reconstruction_error < 0.1:
            score += 0.3

        # Check for pattern regularity (high R-squared indicates artificial pattern)
        if features.temporal.trend_r_squared > 0.8:
            score += 0.3

        # Check if anomaly is well-isolated (single sensor affected)
        if features.correlation.anomaly_isolation_score > 0.7:
            score += 0.2

        # Check for consistency in error pattern
        if features.residual.error_variance < 0.01:
            score += 0.2

        return min(score, 1.0)

    def _check_temporal_manipulation(self, features: AllFeatures) -> float:
        """
        Check for temporal manipulation attacks.

        Temporal attacks show:
        - Irregular time series patterns
        - Sudden spikes in error
        - Temporal inconsistencies
        """
        score = 0.0

        # Check for high change point score
        if features.temporal.change_point_score > 1.0:
            score += 0.4

        # Check for high volatility
        if features.temporal.volatility_index > 2.0:
            score += 0.3

        # Check for low trend R-squared (irregular pattern)
        if features.temporal.trend_r_squared < 0.3:
            score += 0.3

        return min(score, 1.0)

    def _check_multi_sensor_corruption(self, features: AllFeatures) -> float:
        """
        Check for multi-sensor corruption attacks.

        Multi-sensor attacks show:
        - Multiple sensors affected
        - Correlated anomalies
        - Low inter-sensor correlation
        """
        score = 0.0

        # Check if multiple sensors affected
        if features.correlation.correlation_breakdown_score > 0.6:
            score += 0.4

        # Check for low inter-sensor correlation
        if features.correlation.inter_sensor_correlation < 0.3:
            score += 0.3

        # Check if anomaly is NOT isolated
        if features.correlation.anomaly_isolation_score < 0.5:
            score += 0.3

        return min(score, 1.0)

    def _check_jamming(self, features: AllFeatures) -> float:
        """
        Check for jamming attacks.

        Jamming shows:
        - High error variance
        - Low stability
        - Persistent anomaly
        """
        score = 0.0

        # Check for low stability
        if features.temporal.stability_score < 0.3:
            score += 0.4

        # Check for high volatility
        if features.temporal.volatility_index > 3.0:
            score += 0.3

        # Check for persistent anomaly (low variance in error indicates persistence)
        if features.residual.error_variance < 0.1:
            score += 0.3

        return min(score, 1.0)

    def _check_sensor_drift(self, features: AllFeatures) -> float:
        """
        Check for sensor drift.

        Drift shows:
        - High trend slope
        - Good R-squared fit (indicates gradual change)
        - High correlation breakdown (sensors drifting separately)
        """
        score = 0.0

        # Check trend R-squared (gradual change)
        if features.temporal.trend_r_squared > self.config.system_drift_thresholds['trend_r_squared']:
            score += 0.3

        # Check for gradual change
        if abs(features.temporal.trend_slope) > self.config.system_drift_thresholds['gradual_change']:
            score += 0.3

        # Check stability (moderate stability indicates drift)
        if features.temporal.stability_score > self.config.system_drift_thresholds['stability_threshold']:
            score += 0.2

        # Check correlation breakdown
        if features.correlation.correlation_breakdown_score > 0.3:
            score += 0.2

        return min(score, 1.0)

    def _check_sensor_stuck(self, features: AllFeatures) -> float:
        """
        Check for stuck sensor.

        Stuck sensor shows:
        - Very low variance
        - High isolation score
        - Low reconstruction error
        """
        score = 0.0

        # Check variance
        if features.consistency.sensor_cross_validation_error < self.config.sensor_failure_thresholds['sensor_variance_threshold']:
            score += 0.4

        # Check if anomaly is well-isolated (single sensor stuck)
        if features.correlation.anomaly_isolation_score > self.config.sensor_failure_thresholds['isolation_score']:
            score += 0.3

        # Check reconstruction error
        if features.residual.reconstruction_error < 0.05:
            score += 0.3

        return min(score, 1.0)

    def _check_sensor_dead(self, features: AllFeatures) -> float:
        """
        Check for dead sensor.

        Dead sensor shows:
        - Very low or zero signal
        - High isolation
        - Complete failure
        """
        score = 0.0

        # Check for complete failure
        if features.residual.prediction_confidence < 0.1:
            score += 0.5

        # Check if well-isolated
        if features.correlation.anomaly_isolation_score > 0.8:
            score += 0.3

        # Check for persistent pattern (dead sensor = no change)
        if features.temporal.stability_score > 0.9:
            score += 0.2

        return min(score, 1.0)

    def _check_communication_loss(self, features: AllFeatures) -> float:
        """
        Check for communication loss.

        Communication loss shows:
        - Multiple sensors affected
        - Persistent anomaly
        - Correlation breakdown
        """
        score = 0.0

        # Check if multiple sensors affected
        affected_features = 0
        if features.consistency.gps_velocity_mismatch > 0.3:
            affected_features += 1
        if features.consistency.sensor_cross_validation_error > 0.5:
            affected_features += 1
        if features.correlation.inter_sensor_correlation < 0.4:
            affected_features += 1

        if affected_features >= 2:
            score += 0.4

        # Check for persistent pattern
        if features.temporal.stability_score > self.config.communication_loss_thresholds['persistent_pattern']:
            score += 0.3

        # Check correlation breakdown
        if features.correlation.correlation_breakdown_score > self.config.communication_loss_thresholds['correlation_breakdown']:
            score += 0.3

        return min(score, 1.0)

    def _check_system_degradation(self, features: AllFeatures) -> float:
        """
        Check for general system degradation.

        Degradation shows:
        - Multiple issues
        - Moderate severity
        - Gradual onset
        """
        score = 0.0

        # Check for moderate stability (not stuck, not perfect)
        if 0.3 < features.temporal.stability_score < 0.7:
            score += 0.3

        # Check for correlation breakdown
        if features.correlation.correlation_breakdown_score > 0.3:
            score += 0.3

        # Check residual error pattern
        if 0.1 < features.residual.mean_absolute_error < 1.0:
            score += 0.2

        # Check prediction confidence (not completely failed)
        if 0.3 < features.residual.prediction_confidence < 0.8:
            score += 0.2

        return min(score, 1.0)

    def _initialize_threat_signatures(self) -> Dict[str, Dict]:
        """
        Initialize threat signatures for pattern matching.

        Returns:
            Dictionary of threat signatures
        """
        return {
            'gps_spoofing': {
                'primary_features': ['gps_velocity_mismatch', 'position_velocity_correlation'],
                'secondary_features': ['acceleration_consistency', 'change_point_score'],
                'min_score': 0.5
            },
            'sensor_injection': {
                'primary_features': ['reconstruction_error', 'trend_r_squared'],
                'secondary_features': ['anomaly_isolation_score', 'error_variance'],
                'min_score': 0.5
            },
            'sensor_drift': {
                'primary_features': ['trend_r_squared', 'trend_slope'],
                'secondary_features': ['stability_score', 'correlation_breakdown_score'],
                'min_score': 0.5
            },
            'sensor_stuck': {
                'primary_features': ['sensor_cross_validation_error', 'anomaly_isolation_score'],
                'secondary_features': ['reconstruction_error'],
                'min_score': 0.5
            },
            'communication_loss': {
                'primary_features': ['correlation_breakdown_score', 'stability_score'],
                'secondary_features': ['inter_sensor_correlation'],
                'min_score': 0.5
            }
        }

    def _get_attack_type_from_scores(
        self,
        gps_score: float,
        injection_score: float,
        manipulation_score: float,
        corruption_score: float,
        jamming_score: float
    ) -> str:
        """Get attack type from scores"""
        scores = {
            'gps_spoofing': gps_score,
            'sensor_injection': injection_score,
            'temporal_manipulation': manipulation_score,
            'multi_sensor_corruption': corruption_score,
            'jamming': jamming_score
        }
        return max(scores, key=scores.get)

    def _get_failure_type_from_scores(
        self,
        drift_score: float,
        stuck_score: float,
        dead_score: float,
        comm_loss_score: float,
        degradation_score: float
    ) -> str:
        """Get failure type from scores"""
        scores = {
            'sensor_drift': drift_score,
            'sensor_stuck': stuck_score,
            'sensor_dead': dead_score,
            'communication_loss': comm_loss_score,
            'system_degradation': degradation_score
        }
        return max(scores, key=scores.get)

    def _generate_attack_reasoning(
        self,
        features: AllFeatures,
        attack_type: str,
        triggered_rules: List[str]
    ) -> str:
        """Generate natural language reasoning for attack classification"""
        reasoning_parts = [f"ATTACK CLASSIFICATION: {attack_type.upper()}"]

        reasoning_parts.append(f"\nConfidence: High (based on {len(triggered_rules)} rule matches)")

        reasoning_parts.append("\nKey Indicators:")

        # Add specific feature-based reasoning based on attack type
        if attack_type == "gps_spoofing":
            reasoning_parts.append(
                f"  • GPS/velocity mismatch: {features.consistency.gps_velocity_mismatch:.3f} "
                f"(threshold: {self.config.gps_spoofing_thresholds['gps_velocity_mismatch']})"
            )
            reasoning_parts.append(
                f"  • Position-velocity correlation: {features.consistency.position_velocity_correlation:.3f} "
                f"(threshold: {self.config.gps_spoofing_thresholds['position_velocity_correlation']})"
            )

        elif attack_type == "sensor_injection":
            reasoning_parts.append(
                f"  • Signal pattern regularity: {features.temporal.trend_r_squared:.3f}"
            )
            reasoning_parts.append(
                f"  • Anomaly isolation: {features.correlation.anomaly_isolation_score:.3f}"
            )

        # Add temporal context
        if features.temporal.change_point_score > 1.0:
            reasoning_parts.append(
                f"\nPattern: Sudden onset (change point score: {features.temporal.change_point_score:.2f})"
            )

        reasoning_parts.append(
            f"\nInterpretation: This anomaly exhibits characteristics consistent with "
            f"a {attack_type.replace('_', ' ')} attack. The triggered rules ({', '.join(triggered_rules)}) "
            f"indicate adversarial activity rather than natural system failure."
        )

        reasoning_parts.append(
            f"\nRecommended Actions:\n"
            f"  1. Verify signal authenticity\n"
            f"  2. Check for external interference\n"
            f"  3. Consider switching to backup sensors\n"
            f"  4. Alert security team"
        )

        return "\n".join(reasoning_parts)

    def _generate_failure_reasoning(
        self,
        features: AllFeatures,
        failure_type: str,
        triggered_rules: List[str]
    ) -> str:
        """Generate natural language reasoning for failure classification"""
        reasoning_parts = [f"FAILURE CLASSIFICATION: {failure_type.upper()}"]

        reasoning_parts.append(f"\nConfidence: Moderate (based on {len(triggered_rules)} rule matches)")

        reasoning_parts.append("\nKey Indicators:")

        # Add specific feature-based reasoning
        if failure_type == "sensor_drift":
            reasoning_parts.append(
                f"  • Trend R-squared: {features.temporal.trend_r_squared:.3f}"
            )
            reasoning_parts.append(
                f"  • Trend slope: {features.temporal.trend_slope:.4f}"
            )

        elif failure_type == "sensor_stuck":
            reasoning_parts.append(
                f"  • Sensor variance: {features.consistency.sensor_cross_validation_error:.3f}"
            )
            reasoning_parts.append(
                f"  • Isolation score: {features.correlation.anomaly_isolation_score:.3f}"
            )

        # Add temporal context
        if features.temporal.stability_score > 0.5:
            reasoning_parts.append(
                f"\nPattern: Persistent deviation (stability: {features.temporal.stability_score:.3f})"
            )

        reasoning_parts.append(
            f"\nInterpretation: This anomaly exhibits characteristics consistent with "
            f"a {failure_type.replace('_', ' ')}. The triggered rules ({', '.join(triggered_rules)}) "
            f"indicate natural system degradation rather than adversarial activity."
        )

        reasoning_parts.append(
            f"\nRecommended Actions:\n"
            f"  1. Schedule maintenance\n"
            f"  2. Recalibrate affected sensors\n"
            f"  3. Check for environmental factors\n"
            f"  4. Update maintenance logs"
        )

        return "\n".join(reasoning_parts)

    def batch_classify(
        self,
        features_list: List[AllFeatures]
    ) -> List[RuleBasedResult]:
        """
        Classify multiple anomalies in batch.

        Args:
            features_list: List of AllFeatures objects

        Returns:
            List of RuleBasedResult objects
        """
        results = []
        for features in features_list:
            try:
                result = self.classify(features)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying features: {e}")
                # Return unknown on error
                results.append(RuleBasedResult(
                    threat_type=ThreatType.UNKNOWN,
                    specific_type=None,
                    confidence=0.0,
                    reasoning=f"Classification error: {str(e)}",
                    triggered_rules=[],
                    risk_score=0.0
                ))

        return results