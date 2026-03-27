"""
Explanation Templates

Rule-based explanation templates for different threat types.
Generates natural language explanations for threat classifications.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from .temporal_analyzer import TemporalAnalysis, TemporalPattern
from .rule_based_heuristics import ThreatType, AttackType, FailureType

logger = logging.getLogger(__name__)


@dataclass
class TemplateContext:
    """Context for template generation"""
    channel_id: str
    threat_type: str
    confidence: float
    risk_score: float
    top_features: List[Dict]
    temporal_analysis: Optional[TemporalAnalysis] = None
    specific_type: Optional[str] = None
    triggered_rules: Optional[List[str]] = None


class ExplanationTemplates:
    """
    Generates natural language explanations for threat classifications.

    Uses templates to create human-readable explanations based on threat type,
    features, and temporal patterns.
    """

    def __init__(self, detail_level: str = "detailed"):
        """
        Initialize explanation templates.

        Args:
            detail_level: Level of detail ('simple', 'detailed', 'technical')
        """
        self.detail_level = detail_level

    def generate_explanation(self, context: TemplateContext) -> str:
        """
        Generate explanation based on context.

        Args:
            context: Template context

        Returns:
            Natural language explanation
        """
        # Route to appropriate template based on threat type
        if context.threat_type == "attack":
            return self._generate_attack_explanation(context)
        elif context.threat_type == "failure":
            return self._generate_failure_explanation(context)
        else:
            return self._generate_unknown_explanation(context)

    def _generate_attack_explanation(self, context: TemplateContext) -> str:
        """Generate explanation for attack classification"""
        if context.specific_type == "gps_spoofing":
            return self._generate_gps_spoofing_explanation(context)
        elif context.specific_type == "sensor_injection":
            return self._generate_sensor_injection_explanation(context)
        elif context.specific_type == "temporal_manipulation":
            return self._generate_temporal_manipulation_explanation(context)
        elif context.specific_type == "multi_sensor_corruption":
            return self._generate_multi_sensor_corruption_explanation(context)
        elif context.specific_type == "jamming":
            return self._generate_jamming_explanation(context)
        else:
            return self._generate_generic_attack_explanation(context)

    def _generate_failure_explanation(self, context: TemplateContext) -> str:
        """Generate explanation for failure classification"""
        if context.specific_type == "sensor_drift":
            return self._generate_sensor_drift_explanation(context)
        elif context.specific_type == "sensor_stuck":
            return self._generate_sensor_stuck_explanation(context)
        elif context.specific_type == "sensor_dead":
            return self._generate_sensor_dead_explanation(context)
        elif context.specific_type == "communication_loss":
            return self._generate_communication_loss_explanation(context)
        elif context.specific_type == "system_degradation":
            return self._generate_system_degradation_explanation(context)
        else:
            return self._generate_generic_failure_explanation(context)

    def _generate_gps_spoofing_explanation(self, context: TemplateContext) -> str:
        """Generate GPS spoofing explanation"""
        parts = []

        # Header
        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: GPS SPOOFING ATTACK")
        parts.append("=" * 80)

        # Confidence and risk
        parts.append(f"\nConfidence Level: {context.confidence:.1%}")
        parts.append(f"Risk Score: {context.risk_score:.2f}/1.0")
        parts.append(f"Channel: {context.channel_id}")

        # Temporal pattern
        if context.temporal_analysis:
            parts.append(f"\nTemporal Pattern: {context.temporal_analysis.pattern_type.value.upper()}")
            parts.append(f"Duration: {context.temporal_analysis.duration} timesteps")
            parts.append(f"Intensity: {context.temporal_analysis.intensity:.4f}")

        # Key indicators
        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        # GPS-specific indicators
        gps_mismatch = self._get_feature_value(context.top_features, 'gps_velocity_mismatch')
        if gps_mismatch:
            parts.append(f"  • GPS/Velocity Mismatch: {gps_mismatch:.3f}")
            parts.append(f"    → Indicates position and velocity are inconsistent")

        pos_vel_corr = self._get_feature_value(context.top_features, 'position_velocity_correlation')
        if pos_vel_corr:
            parts.append(f"  • Position-Velocity Correlation: {pos_vel_corr:.3f}")
            parts.append(f"    → Lower values suggest artificial signal patterns")

        accel_consistency = self._get_feature_value(context.top_features, 'acceleration_consistency')
        if accel_consistency:
            parts.append(f"  • Acceleration Consistency: {accel_consistency:.3f}")
            parts.append(f"    → Inconsistent with expected motion dynamics")

        change_point = self._get_feature_value(context.top_features, 'change_point_score')
        if change_point:
            parts.append(f"  • Change Point Score: {change_point:.3f}")
            parts.append(f"    → Sudden onset characteristic of spoofing")

        # Interpretation
        parts.append("\n" + "-" * 80)
        parts.append("INTERPRETATION:")
        parts.append("-" * 80)
        parts.append(
            "This anomaly exhibits characteristics consistent with a GPS spoofing attack. "
            "The satellite signals appear to be artificially generated or replayed, causing "
            "position and velocity measurements that are inconsistent with the expected "
            "motion profile of the vehicle."
        )

        if context.temporal_analysis:
            if context.temporal_analysis.pattern_type == TemporalPattern.SPIKE:
                parts.append(
                    "The SPIKE pattern indicates a sudden onset, which is typical of "
                    "targeted spoofing attacks."
                )
            elif context.temporal_analysis.pattern_type == TemporalPattern.INTERMITTENT:
                parts.append(
                    "The INTERMITTENT pattern suggests intermittent interference or "
                    "selective spoofing of specific signals."
                )

        # Recommended actions
        parts.append("\n" + "-" * 80)
        parts.append("RECOMMENDED ACTIONS:")
        parts.append("-" * 80)
        parts.append("  1. IMMEDIATE: Switch to backup navigation systems")
        parts.append("  2. VERIFY: Cross-check position with inertial navigation")
        parts.append("  3. MONITOR: Track signal quality and consistency")
        parts.append("  4. ALERT: Notify security operations center")
        parts.append("  5. DOCUMENT: Log all anomalies for forensic analysis")

        # Severity assessment
        parts.append("\n" + "-" * 80)
        parts.append("SEVERITY ASSESSMENT:")
        parts.append("-" * 80)

        if context.confidence > 0.9:
            parts.append("CRITICAL: High confidence GPS spoofing detected. Navigation integrity compromised.")
        elif context.confidence > 0.7:
            parts.append("HIGH: Probable GPS spoofing. Verification recommended.")
        else:
            parts.append("MEDIUM: Potential GPS anomalies. Monitor closely.")

        return "\n".join(parts)

    def _generate_sensor_injection_explanation(self, context: TemplateContext) -> str:
        """Generate sensor injection explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: SENSOR INJECTION ATTACK")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")
        parts.append(f"Risk Score: {context.risk_score:.2f}/1.0")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        reconstruction_error = self._get_feature_value(context.top_features, 'reconstruction_error')
        if reconstruction_error:
            parts.append(f"  • Reconstruction Error: {reconstruction_error:.3f}")
            parts.append(f"    → Low error suggests artificial signal patterns")

        trend_r2 = self._get_feature_value(context.top_features, 'trend_r_squared')
        if trend_r2:
            parts.append(f"  • Trend R-squared: {trend_r2:.3f}")
            parts.append(f"    → High regularity indicates synthetic data")

        isolation_score = self._get_feature_value(context.top_features, 'anomaly_isolation_score')
        if isolation_score:
            parts.append(f"  • Anomaly Isolation: {isolation_score:.3f}")
            parts.append(f"    → Single sensor affected")

        parts.append("\nINTERPRETATION:")
        parts.append(
            "This anomaly exhibits characteristics of a sensor injection attack. "
            "The injected signals show artificial regularity and low reconstruction error, "
            "indicating they do not follow natural sensor behavior patterns."
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Cross-validate with redundant sensors")
        parts.append("  2. Check for unauthorized access to sensor networks")
        parts.append("  3. Monitor for continued injection attempts")
        parts.append("  4. Implement sensor authentication protocols")

        return "\n".join(parts)

    def _generate_temporal_manipulation_explanation(self, context: TemplateContext) -> str:
        """Generate temporal manipulation explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: TEMPORAL MANIPULATION ATTACK")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        change_point = self._get_feature_value(context.top_features, 'change_point_score')
        if change_point:
            parts.append(f"  • Change Point Score: {change_point:.3f}")
            parts.append(f"    → Sudden discontinuities in time series")

        volatility = self._get_feature_value(context.top_features, 'volatility_index')
        if volatility:
            parts.append(f"  • Volatility Index: {volatility:.3f}")
            parts.append(f"    → Irregular temporal patterns")

        trend_r2 = self._get_feature_value(context.top_features, 'trend_r_squared')
        if trend_r2:
            parts.append(f"  • Trend R-squared: {trend_r2:.3f}")
            parts.append(f"    → Low predictability")

        parts.append("\nINTERPRETATION:")
        parts.append(
            "The time series exhibits irregular patterns inconsistent with natural behavior. "
            "This suggests temporal manipulation of the signal timestamps or data ordering."
        )

        return "\n".join(parts)

    def _generate_multi_sensor_corruption_explanation(self, context: TemplateContext) -> str:
        """Generate multi-sensor corruption explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: MULTI-SENSOR CORRUPTION ATTACK")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        corr_breakdown = self._get_feature_value(context.top_features, 'correlation_breakdown_score')
        if corr_breakdown:
            parts.append(f"  • Correlation Breakdown: {corr_breakdown:.3f}")
            parts.append(f"    → Multiple sensors showing correlated anomalies")

        inter_sensor_corr = self._get_feature_value(context.top_features, 'inter_sensor_correlation')
        if inter_sensor_corr:
            parts.append(f"  • Inter-Sensor Correlation: {inter_sensor_corr:.3f}")
            parts.append(f"    → Unusually low correlation between sensors")

        isolation_score = self._get_feature_value(context.top_features, 'anomaly_isolation_score')
        if isolation_score:
            parts.append(f"  • Anomaly Isolation: {isolation_score:.3f}")
            parts.append(f"    → Widespread sensor corruption")

        return "\n".join(parts)

    def _generate_jamming_explanation(self, context: TemplateContext) -> str:
        """Generate jamming explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: JAMMING ATTACK")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        stability = self._get_feature_value(context.top_features, 'stability_score')
        if stability:
            parts.append(f"  • Stability Score: {stability:.3f}")
            parts.append(f"    → Low stability indicates interference")

        volatility = self._get_feature_value(context.top_features, 'volatility_index')
        if volatility:
            parts.append(f"  • Volatility Index: {volatility:.3f}")
            parts.append(f"    → High volatility from jamming signals")

        return "\n".join(parts)

    def _generate_sensor_drift_explanation(self, context: TemplateContext) -> str:
        """Generate sensor drift explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: SENSOR DRIFT")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")
        parts.append(f"Risk Score: {context.risk_score:.2f}/1.0")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        trend_r2 = self._get_feature_value(context.top_features, 'trend_r_squared')
        if trend_r2:
            parts.append(f"  • Trend R-squared: {trend_r2:.3f}")
            parts.append(f"    → Strong linear trend indicates gradual drift")

        trend_slope = self._get_feature_value(context.top_features, 'trend_slope')
        if trend_slope:
            parts.append(f"  • Trend Slope: {trend_slope:.4f}")
            parts.append(f"    → Rate of gradual change")

        stability = self._get_feature_value(context.top_features, 'stability_score')
        if stability:
            parts.append(f"  • Stability Score: {stability:.3f}")
            parts.append(f"    → Moderate stability consistent with drift")

        parts.append("\nINTERPRETATION:")
        parts.append(
            "This anomaly exhibits characteristics of sensor drift - a gradual deviation "
            "from expected values over time. The linear trend pattern and moderate R-squared "
            "value indicate systematic degradation rather than sudden failure."
        )

        if context.temporal_analysis:
            parts.append(
                f"The {context.temporal_analysis.pattern_type.value} pattern confirms "
                f"gradual onset typical of sensor drift."
            )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Schedule sensor recalibration")
        parts.append("  2. Check for environmental factors (temperature, humidity)")
        parts.append("  3. Review maintenance schedule")
        parts.append("  4. Monitor trend continuation")
        parts.append("  5. Plan sensor replacement if drift continues")

        return "\n".join(parts)

    def _generate_sensor_stuck_explanation(self, context: TemplateContext) -> str:
        """Generate sensor stuck explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: SENSOR STUCK")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        sensor_variance = self._get_feature_value(context.top_features, 'sensor_cross_validation_error')
        if sensor_variance:
            parts.append(f"  • Sensor Variance: {sensor_variance:.3f}")
            parts.append(f"    → Very low variance indicates stuck values")

        isolation_score = self._get_feature_value(context.top_features, 'anomaly_isolation_score')
        if isolation_score:
            parts.append(f"  • Anomaly Isolation: {isolation_score:.3f}")
            parts.append(f"    → Single sensor affected")

        reconstruction_error = self._get_feature_value(context.top_features, 'reconstruction_error')
        if reconstruction_error:
            parts.append(f"  • Reconstruction Error: {reconstruction_error:.3f}")
            parts.append(f"    → Low error from consistent stuck values")

        return "\n".join(parts)

    def _generate_sensor_dead_explanation(self, context: TemplateContext) -> str:
        """Generate sensor dead explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: SENSOR DEAD")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        prediction_confidence = self._get_feature_value(context.top_features, 'prediction_confidence')
        if prediction_confidence:
            parts.append(f"  • Prediction Confidence: {prediction_confidence:.3f}")
            parts.append(f"    → Very low confidence indicates complete failure")

        isolation_score = self._get_feature_value(context.top_features, 'anomaly_isolation_score')
        if isolation_score:
            parts.append(f"  • Anomaly Isolation: {isolation_score:.3f}")
            parts.append(f"    → Single sensor completely failed")

        stability = self._get_feature_value(context.top_features, 'stability_score')
        if stability:
            parts.append(f"  • Stability Score: {stability:.3f}")
            parts.append(f"    → Perfect stability (no signal)")

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. IMMEDIATE: Switch to redundant sensor")
        parts.append("  2. Replace sensor hardware")
        parts.append("  3. Check power and connections")
        parts.append("  4. Update maintenance records")

        return "\n".join(parts)

    def _generate_communication_loss_explanation(self, context: TemplateContext) -> str:
        """Generate communication loss explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: COMMUNICATION LOSS")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        corr_breakdown = self._get_feature_value(context.top_features, 'correlation_breakdown_score')
        if corr_breakdown:
            parts.append(f"  • Correlation Breakdown: {corr_breakdown:.3f}")
            parts.append(f"    → Multiple sensors affected simultaneously")

        stability = self._get_feature_value(context.top_features, 'stability_score')
        if stability:
            parts.append(f"  • Stability Score: {stability:.3f}")
            parts.append(f"    → Persistent pattern indicates communication issue")

        inter_sensor_corr = self._get_feature_value(context.top_features, 'inter_sensor_correlation')
        if inter_sensor_corr:
            parts.append(f"  • Inter-Sensor Correlation: {inter_sensor_corr:.3f}")
            parts.append(f"    → Breakdown in normal sensor relationships")

        parts.append("\nINTERPRETATION:")
        parts.append(
            "Multiple sensors are showing anomalous behavior simultaneously with a "
            "persistent pattern. This is characteristic of communication loss rather "
            "than individual sensor failures."
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Check network connectivity")
        parts.append("  2. Verify data link status")
        parts.append("  3. Inspect physical connections")
        parts.append("  4. Check for electromagnetic interference")
        parts.append("  5. Switch to backup communication channels")

        return "\n".join(parts)

    def _generate_system_degradation_explanation(self, context: TemplateContext) -> str:
        """Generate system degradation explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: SYSTEM DEGRADATION")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("KEY INDICATORS:")
        parts.append("-" * 80)

        stability = self._get_feature_value(context.top_features, 'stability_score')
        if stability:
            parts.append(f"  • Stability Score: {stability:.3f}")
            parts.append(f"    → Moderate stability indicates partial degradation")

        corr_breakdown = self._get_feature_value(context.top_features, 'correlation_breakdown_score')
        if corr_breakdown:
            parts.append(f"  • Correlation Breakdown: {corr_breakdown:.3f}")
            parts.append(f"    → Multiple systems affected")

        mean_error = self._get_feature_value(context.top_features, 'mean_absolute_error')
        if mean_error:
            parts.append(f"  • Mean Absolute Error: {mean_error:.3f}")
            parts.append(f"    → Moderate error level")

        prediction_confidence = self._get_feature_value(context.top_features, 'prediction_confidence')
        if prediction_confidence:
            parts.append(f"  • Prediction Confidence: {prediction_confidence:.3f}")
            parts.append(f"    → Partial system capability")

        parts.append("\nINTERPRETATION:")
        parts.append(
            "The system is experiencing gradual degradation across multiple components. "
            "The moderate stability and error levels suggest aging hardware or "
            "environmental factors affecting overall system performance."
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Schedule comprehensive system maintenance")
        parts.append("  2. Check environmental conditions")
        parts.append("  3. Review system usage patterns")
        parts.append("  4. Plan component replacement schedule")
        parts.append("  5. Update monitoring thresholds")

        return "\n".join(parts)

    def _generate_generic_attack_explanation(self, context: TemplateContext) -> str:
        """Generate generic attack explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"THREAT DETECTED: ADVERSARIAL ACTIVITY")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")
        parts.append(f"Risk Score: {context.risk_score:.2f}/1.0")

        parts.append("\n" + "-" * 80)
        parts.append("ANALYSIS:")
        parts.append("-" * 80)
        parts.append(
            "The anomaly patterns detected do not match known natural failure modes. "
            "The characteristics suggest adversarial activity or intentional interference "
            "with system operations."
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Activate security protocols")
        parts.append("  2. Switch to backup systems")
        parts.append("  3. Alert security team")
        parts.append("  4. Begin forensic analysis")

        return "\n".join(parts)

    def _generate_generic_failure_explanation(self, context: TemplateContext) -> str:
        """Generate generic failure explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"FAILURE DETECTED: SYSTEM MALFUNCTION")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("ANALYSIS:")
        parts.append("-" * 80)
        parts.append(
            "The anomaly patterns are consistent with natural system failure or "
            "degradation. No indications of adversarial activity detected."
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Schedule maintenance")
        parts.append("  2. Check system logs")
        parts.append("  3. Plan component replacement")

        return "\n".join(parts)

    def _generate_unknown_explanation(self, context: TemplateContext) -> str:
        """Generate unknown classification explanation"""
        parts = []

        parts.append("=" * 80)
        parts.append(f"ANOMALY DETECTED: UNKNOWN PATTERN")
        parts.append("=" * 80)

        parts.append(f"\nConfidence Level: {context.confidence:.1%}")

        parts.append("\n" + "-" * 80)
        parts.append("ANALYSIS:")
        parts.append("-" * 80)
        parts.append(
            "The detected anomaly does not match known patterns for attacks or failures. "
            "Classification confidence is below threshold. This could indicate:\n"
            "  • New, previously unseen threat type\n"
            "  • Data quality issue\n"
            "  • System calibration problem"
        )

        parts.append("\nRECOMMENDED ACTIONS:")
        parts.append("  1. Review raw data quality")
        parts.append("  2. Check system calibration")
        parts.append("  3. Monitor for pattern evolution")
        parts.append("  4. Consider manual review")

        return "\n".join(parts)

    def _get_feature_value(self, top_features: List[Dict], feature_name: str) -> Optional[float]:
        """Get feature value by name"""
        for feature in top_features:
            if feature.get('name') == feature_name:
                return feature.get('value')
        return None

    def generate_summary(self, context: TemplateContext) -> str:
        """
        Generate brief summary explanation.

        Args:
            context: Template context

        Returns:
            Brief summary string
        """
        if context.threat_type == "attack":
            return f"Attack detected ({context.specific_type or 'adversarial'}) - Confidence: {context.confidence:.1%}"
        elif context.threat_type == "failure":
            return f"System failure ({context.specific_type or 'malfunction'}) - Confidence: {context.confidence:.1%}"
        else:
            return f"Unknown anomaly - Confidence: {context.confidence:.1%}"