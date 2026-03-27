"""
Explainability Engine

Main orchestrator for threat explainability.
Coordinates feature attribution, temporal analysis, and template generation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import time
from datetime import datetime

from app.ai.ThreatClassificationEngine.helpers.temporal_analyzer import TemporalPatternDetector, TemporalAnalysis, TemporalPattern
from .explanation_templates import ExplanationTemplates, TemplateContext
from app.ai.ThreatClassificationEngine.config.threat_config import ExplainabilityConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityResult:
    """Result of explainability analysis"""
    # Core explanation
    top_features: List[Dict]  # List of feature dicts with name, importance, value
    reason: str  # Natural language explanation
    pattern_type: str  # spike, drift, persistent, intermittent
    confidence_explanation: float  # [0, 1] confidence in explanation

    # Temporal context
    temporal_analysis: TemporalAnalysis
    anomaly_window: Tuple[int, int]

    # Metadata
    attribution_method: str
    processing_time_ms: float
    timestamp: datetime


class ExplainabilityEngine:
    """
    Main explainability orchestrator.

    Combines feature attribution, temporal pattern detection, and template
    generation to create comprehensive, human-readable explanations for
    threat classifications.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfig] = None,
        lstm_model=None
    ):
        """
        Initialize explainability engine.

        Args:
            config: Configuration (uses default if not provided)
            lstm_model: Optional LSTM model for feature attribution
        """
        self.config = config or get_config().explainability
        self.lstm_model = lstm_model

        # Initialize components
        self.temporal_detector = TemporalPatternDetector(
            sensitivity=self.config.pattern_detection_sensitivity
        )
        self.templates = ExplanationTemplates(
            detail_level=self.config.template_detail_level
        )

        # Performance tracking
        self.processing_times = {
            'temporal_analysis': [],
            'feature_attribution': [],
            'template_generation': [],
            'total': []
        }

        # Simple LRU cache for performance
        self._cache = {}
        self._cache_size = self.config.cache_size if self.config.cache_attributions else 0

        logger.info("Explainability engine initialized")

    def explain(
        self,
        classification_result: Any,
        features: Any,
        anomaly_window: Tuple[int, int],
        error_signal: Optional[List[float]] = None,
        detection_threshold: Optional[float] = None,
        channel_id: str = "unknown"
    ) -> ExplainabilityResult:
        """
        Generate complete explanation for threat classification.

        Args:
            classification_result: ThreatClassification result
            features: AllFeatures object or feature dict
            anomaly_window: (start, end) indices of anomaly
            error_signal: Optional error signal for temporal analysis
            detection_threshold: Optional detection threshold
            channel_id: Channel identifier

        Returns:
            ExplainabilityResult with complete explanation
        """
        start_time = time.time()

        try:
            # Step 1: Extract features
            feature_names, feature_values = self._extract_features(features)

            # Step 2: Temporal analysis
            temporal_start = time.time()
            temporal_analysis = self._perform_temporal_analysis(
                error_signal, anomaly_window, detection_threshold
            )
            self.processing_times['temporal_analysis'].append(
                (time.time() - temporal_start) * 1000
            )

            # Step 3: Feature attribution
            attribution_start = time.time()
            top_features = self._compute_feature_attribution(
                feature_names, feature_values, classification_result
            )
            self.processing_times['feature_attribution'].append(
                (time.time() - attribution_start) * 1000
            )

            # Step 4: Generate natural language explanation
            template_start = time.time()
            reason = self._generate_explanation(
                classification_result, top_features, temporal_analysis, channel_id
            )
            self.processing_times['template_generation'].append(
                (time.time() - template_start) * 1000
            )

            # Step 5: Compute overall confidence
            confidence = self._compute_explanation_confidence(
                top_features, temporal_analysis
            )

            # Total processing time
            total_time = (time.time() - start_time) * 1000
            self.processing_times['total'].append(total_time)

            result = ExplainabilityResult(
                top_features=top_features,
                reason=reason,
                pattern_type=temporal_analysis.pattern_type.value,
                confidence_explanation=confidence,
                temporal_analysis=temporal_analysis,
                anomaly_window=anomaly_window,
                attribution_method=self.config.attribution_method,
                processing_time_ms=total_time,
                timestamp=datetime.now()
            )

            logger.debug(
                f"Generated explanation for {channel_id}: "
                f"{temporal_analysis.pattern_type.value} "
                f"({total_time:.2f}ms)"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating explanation: {e}", exc_info=True)
            raise

    def _extract_features(
        self,
        features: Any
    ) -> Tuple[List[str], List[float]]:
        """
        Extract feature names and values.

        Args:
            features: AllFeatures object or feature dict

        Returns:
            Tuple of (feature_names, feature_values)
        """
        # Handle AllFeatures object
        if hasattr(features, 'to_array') and hasattr(features, 'get_feature_names'):
            feature_names = features.get_feature_names()
            feature_values = features.to_array().tolist()
        # Handle dict
        elif isinstance(features, dict):
            if 'feature_names' in features and 'feature_values' in features:
                feature_names = features['feature_names']
                feature_values = features['feature_values']
            else:
                # Assume flat dict
                feature_names = list(features.keys())
                feature_values = list(features.values())
        # Handle list/tuple
        elif isinstance(features, (list, tuple)):
            feature_names = [f"feature_{i}" for i in range(len(features))]
            feature_values = list(features)
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")

        return feature_names, feature_values

    def _perform_temporal_analysis(
        self,
        error_signal: Optional[List[float]],
        anomaly_window: Tuple[int, int],
        detection_threshold: Optional[float]
    ) -> TemporalAnalysis:
        """
        Perform temporal pattern analysis.

        Args:
            error_signal: Error signal
            anomaly_window: Anomaly window
            detection_threshold: Detection threshold

        Returns:
            TemporalAnalysis result
        """
        if error_signal is None or detection_threshold is None:
            # Return default analysis
            return TemporalAnalysis(
                pattern_type=TemporalPattern.PERSISTENT,
                onset_time=anomaly_window[0],
                duration=anomaly_window[1] - anomaly_window[0],
                intensity=0.0,
                trend_slope=0.0,
                stability_score=0.5,
                volatility_index=0.0,
                change_point_score=0.0,
                monotonic_score=0.5
            )

        # Ensure error_signal is numpy array
        import numpy as np
        error_array = np.array(error_signal)

        # Perform analysis
        analysis = self.temporal_detector.analyze_pattern(
            error_array, anomaly_window, detection_threshold
        )

        return analysis

    def _compute_feature_attribution(
        self,
        feature_names: List[str],
        feature_values: List[float],
        classification_result: Any
    ) -> List[Dict]:
        """
        Compute feature attribution scores.

        Args:
            feature_names: List of feature names
            feature_values: List of feature values
            classification_result: Classification result

        Returns:
            List of feature dicts with attribution scores
        """
        # For now, use simplified attribution based on feature values
        # In real implementation, would use Fast Integrated Gradients

        # Calculate simple importance scores based on magnitude and classification
        import numpy as np
        values_array = np.array(feature_values)

        # Get absolute values for importance
        abs_values = np.abs(values_array)

        # Normalize to [0, 1]
        if np.max(abs_values) > 0:
            normalized_importance = abs_values / np.max(abs_values)
        else:
            normalized_importance = np.zeros_like(abs_values)

        # Adjust based on classification
        if hasattr(classification_result, 'threat_class'):
            if classification_result.threat_class == 'attack':
                # Boost certain features for attacks
                boost_indices = []
                for i, name in enumerate(feature_names):
                    if any(term in name.lower() for term in ['gps', 'position', 'velocity', 'signal']):
                        boost_indices.append(i)

                for idx in boost_indices:
                    if idx < len(normalized_importance):
                        normalized_importance[idx] *= 1.2

            elif classification_result.threat_class == 'failure':
                # Boost sensor-related features for failures
                boost_indices = []
                for i, name in enumerate(feature_names):
                    if any(term in name.lower() for term in ['variance', 'sensor', 'drift', 'stuck']):
                        boost_indices.append(i)

                for idx in boost_indices:
                    if idx < len(normalized_importance):
                        normalized_importance[idx] *= 1.2

        # Cap at 1.0
        normalized_importance = np.minimum(normalized_importance, 1.0)

        # Create feature dicts
        features = []
        for i, (name, value, importance) in enumerate(
            zip(feature_names, feature_values, normalized_importance)
        ):
            features.append({
                'name': name,
                'value': float(value),
                'importance': float(importance),
                'contribution_direction': 'positive' if value > 0 else 'negative'
            })

        # Sort by importance
        features.sort(key=lambda x: x['importance'], reverse=True)

        # Return top features
        return features[:self.config.top_k_features]

    def _generate_explanation(
        self,
        classification_result: Any,
        top_features: List[Dict],
        temporal_analysis: TemporalAnalysis,
        channel_id: str
    ) -> str:
        """
        Generate natural language explanation.

        Args:
            classification_result: Classification result
            top_features: Top contributing features
            temporal_analysis: Temporal analysis
            channel_id: Channel identifier

        Returns:
            Natural language explanation
        """
        # Create template context
        context = TemplateContext(
            channel_id=channel_id,
            threat_type=getattr(classification_result, 'threat_class', 'unknown'),
            confidence=getattr(classification_result, 'confidence', 0.0),
            risk_score=getattr(classification_result, 'risk_score', 0.0),
            top_features=top_features,
            temporal_analysis=temporal_analysis,
            specific_type=getattr(classification_result, 'specific_threat_type', None)
        )

        # Generate explanation
        explanation = self.templates.generate_explanation(context)

        return explanation

    def _compute_explanation_confidence(
        self,
        top_features: List[Dict],
        temporal_analysis: TemporalAnalysis
    ) -> float:
        """
        Compute confidence in explanation.

        Args:
            top_features: Top contributing features
            temporal_analysis: Temporal analysis

        Returns:
            Confidence score [0, 1]
        """
        # Feature confidence
        if top_features:
            # Higher confidence if top feature is clearly dominant
            top_importance = top_features[0]['importance']
            feature_confidence = min(1.0, top_importance * 2)
        else:
            feature_confidence = 0.0

        # Temporal confidence
        # Higher confidence if pattern is clear and stable
        temporal_confidence = temporal_analysis.stability_score

        # Combine (weighted average)
        confidence = 0.6 * feature_confidence + 0.4 * temporal_confidence

        return float(min(1.0, max(0.0, confidence)))

    def batch_explain(
        self,
        classifications: Dict[str, Any],
        features_dict: Dict[str, Any],
        error_signals: Optional[Dict[str, List[float]]] = None,
        detection_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, ExplainabilityResult]:
        """
        Generate explanations for multiple classifications in batch.

        Args:
            classifications: Dict mapping channel_id to classification result
            features_dict: Dict mapping channel_id to features
            error_signals: Optional dict mapping channel_id to error signals
            detection_thresholds: Optional dict mapping channel_id to thresholds

        Returns:
            Dict mapping channel_id to ExplainabilityResult
        """
        results = {}
        error_signals = error_signals or {}
        detection_thresholds = detection_thresholds or {}

        for channel_id, classification in classifications.items():
            try:
                if channel_id not in features_dict:
                    logger.warning(f"No features for channel {channel_id}")
                    continue

                # Get anomaly window (simplified)
                anomaly_window = (0, 100)  # Default window

                explanation = self.explain(
                    classification_result=classification,
                    features=features_dict[channel_id],
                    anomaly_window=anomaly_window,
                    error_signal=error_signals.get(channel_id),
                    detection_threshold=detection_thresholds.get(channel_id),
                    channel_id=channel_id
                )

                results[channel_id] = explanation

            except Exception as e:
                logger.error(f"Error explaining channel {channel_id}: {e}")

                # Add error result
                results[channel_id] = ExplainabilityResult(
                    top_features=[],
                    reason=f"Error generating explanation: {str(e)}",
                    pattern_type="persistent",
                    confidence_explanation=0.0,
                    temporal_analysis=TemporalAnalysis(
                        pattern_type=TemporalPattern.PERSISTENT,
                        onset_time=0,
                        duration=0,
                        intensity=0.0,
                        trend_slope=0.0,
                        stability_score=0.0,
                        volatility_index=0.0,
                        change_point_score=0.0,
                        monotonic_score=0.0
                    ),
                    anomaly_window=(0, 0),
                    attribution_method=self.config.attribution_method,
                    processing_time_ms=0.0,
                    timestamp=datetime.now()
                )

        logger.info(f"Generated {len(results)} explanations")
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

    def get_explainability_info(self) -> Dict[str, Any]:
        """
        Get explainability system information.

        Returns:
            Dictionary with system information
        """
        return {
            'attribution_method': self.config.attribution_method,
            'attribution_steps': self.config.attribution_steps,
            'attribution_baseline': self.config.attribution_baseline,
            'top_k_features': self.config.top_k_features,
            'min_feature_importance': self.config.min_feature_importance,
            'pattern_detection': self.config.detect_patterns,
            'pattern_sensitivity': self.config.pattern_detection_sensitivity,
            'use_templates': self.config.use_templates,
            'template_detail_level': self.config.template_detail_level,
            'cache_enabled': self.config.cache_attributions,
            'cache_size': self.config.cache_size,
            'performance': {
                'target_temporal_ms': 1.0,
                'target_attribution_ms': 10.0,
                'target_total_ms': 15.0
            }
        }

    def clear_cache(self):
        """Clear explanation cache"""
        self._cache.clear()
        logger.info("Explanation cache cleared")