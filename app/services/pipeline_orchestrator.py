"""
Pipeline Orchestrator

Main pipeline coordinator that orchestrates anomaly detection,
threat classification, and explainability in a unified flow.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import asyncio
from datetime import datetime
import numpy as np

from app.ai.threat_classifier import ThreatClassifier, ThreatClassification
from app.ai.explainability_engine import ExplainabilityEngine, ExplainabilityResult
from app.ai.feature_extractor import FeatureExtractor, AllFeatures
from app.config.threat_config import get_config, ThreatClassificationConfig

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Pipeline processing stages"""
    ANOMALY_DETECTION = "anomaly_detection"
    FEATURE_EXTRACTION = "feature_extraction"
    THREAT_CLASSIFICATION = "threat_classification"
    EXPLAINABILITY = "explainability"
    ALERT = "alert"
    BLOCKCHAIN_LOG = "blockchain_log"
    COMPLETE = "complete"


@dataclass
class PipelineEvent:
    """Event in the processing pipeline"""
    stage: PipelineStage
    timestamp: datetime
    channel_id: str
    data: Dict[str, Any]
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete result of pipeline processing"""
    channel_id: str
    anomaly_detected: bool
    anomaly_score: float
    anomaly_window: Optional[Tuple[int, int]] = None
    threat_classification: Optional[ThreatClassification] = None
    explanation: Optional[ExplainabilityResult] = None
    events: List[PipelineEvent] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class PipelineOrchestrator:
    """
    Main orchestrator for the complete anomaly intelligence pipeline.

    Coordinates:
    1. Anomaly Detection (existing TELEMANOM system)
    2. Threat Classification (new)
    3. Explainability (new)
    4. Event Trigger System (alerting)
    5. Blockchain Logging (critical events)
    """

    def __init__(
        self,
        config: Optional[ThreatClassificationConfig] = None,
        threat_classifier: Optional[ThreatClassifier] = None,
        explainability_engine: Optional[ExplainabilityEngine] = None
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Configuration (uses default if not provided)
            threat_classifier: Initialized threat classifier
            explainability_engine: Initialized explainability engine
        """
        self.config = config or get_config()
        self.threat_classifier = threat_classifier
        self.explainability_engine = explainability_engine

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Event handlers
        self.event_handlers = {
            PipelineStage.ALERT: self._handle_alert,
            PipelineStage.BLOCKCHAIN_LOG: self._handle_blockchain_log
        }

        # Performance tracking
        self.processing_times = {
            stage.value: [] for stage in PipelineStage
        }

        # Event history (for debugging/monitoring)
        self.event_history: List[PipelineEvent] = []

        logger.info("Pipeline orchestrator initialized")

    async def process_telemetry(
        self,
        channel_id: str,
        data: np.ndarray,
        anomaly_detector_result: Optional[Any] = None,
        y_true: Optional[np.ndarray] = None,
        y_hat: Optional[np.ndarray] = None,
        error_signal: Optional[np.ndarray] = None
    ) -> PipelineResult:
        """
        Process telemetry through the complete pipeline.

        Args:
            channel_id: Channel identifier
            data: Input telemetry data (timesteps, n_features)
            anomaly_detector_result: Result from AnomalyDetector.run()
            y_true: True values (optional)
            y_hat: Predicted values (optional)
            error_signal: Smoothed error signal (optional)

        Returns:
            PipelineResult with complete analysis
        """
        start_time = time.time()
        events: List[PipelineEvent] = []
        success = True
        error_message = None

        try:
            logger.info(f"Starting pipeline for channel {channel_id}")

            # Stage 1: Anomaly Detection
            if anomaly_detector_result is None:
                # In production, would integrate with existing TELEMANOM detector
                logger.warning(f"No anomaly detection result for {channel_id}, skipping...")
                anomaly_result = None
            else:
                anomaly_result = anomaly_detector_result

            events.append(PipelineEvent(
                stage=PipelineStage.ANOMALY_DETECTION,
                timestamp=datetime.now(),
                channel_id=channel_id,
                data={"anomaly_detected": anomaly_result is not None and len(anomaly_result.anomaly_sequences) > 0},
                processing_time_ms=0.0
            ))

            # If no anomaly, still process for baseline
            threat_classification = None
            explanation = None

            if anomaly_result and len(anomaly_result.anomaly_sequences) > 0:
                # Stage 2: Feature Extraction
                features_start = time.time()
                features = await self._extract_features(
                    anomaly_result, data, y_true, y_hat, error_signal
                )
                features_time = (time.time() - features_start) * 1000

                events.append(PipelineEvent(
                    stage=PipelineStage.FEATURE_EXTRACTION,
                    timestamp=datetime.now(),
                    channel_id=channel_id,
                    data={"n_features": len(features.get_feature_names())},
                    processing_time_ms=features_time
                ))

                # Stage 3: Threat Classification
                if self.threat_classifier is not None:
                    threat_start = time.time()
                    threat_classification = await self._classify_threat(
                        anomaly_result, data, features
                    )
                    threat_time = (time.time() - threat_start) * 1000

                    events.append(PipelineEvent(
                        stage=PipelineStage.THREAT_CLASSIFICATION,
                        timestamp=datetime.now(),
                        channel_id=channel_id,
                        data={
                            "threat_class": threat_classification.threat_class,
                            "confidence": threat_classification.confidence
                        },
                        processing_time_ms=threat_time
                    ))

                    # Stage 4: Explainability
                    if self.explainability_engine is not None:
                        explain_start = time.time()
                        explanation = await self._generate_explanation(
                            threat_classification, features, anomaly_result
                        )
                        explain_time = (time.time() - explain_start) * 1000

                        events.append(PipelineEvent(
                            stage=PipelineStage.EXPLAINABILITY,
                            timestamp=datetime.now(),
                            channel_id=channel_id,
                            data={
                                "pattern_type": explanation.pattern_type,
                                "confidence": explanation.confidence_explanation
                            },
                            processing_time_ms=explain_time
                        ))

                        # Stage 5: Alert Trigger
                        if self._should_trigger_alert(threat_classification):
                            alert_start = time.time()
                            await self._trigger_alert(channel_id, threat_classification, explanation)
                            alert_time = (time.time() - alert_start) * 1000

                            events.append(PipelineEvent(
                                stage=PipelineStage.ALERT,
                                timestamp=datetime.now(),
                                channel_id=channel_id,
                                data={"alert_level": "high" if threat_classification.risk_score > 0.8 else "medium"},
                                processing_time_ms=alert_time
                            ))

                        # Stage 6: Blockchain Log (critical events only)
                        if self._should_log_to_blockchain(threat_classification):
                            blockchain_start = time.time()
                            await self._log_to_blockchain(channel_id, threat_classification, explanation)
                            blockchain_time = (time.time() - blockchain_start) * 1000

                            events.append(PipelineEvent(
                                stage=PipelineStage.BLOCKCHAIN_LOG,
                                timestamp=datetime.now(),
                                channel_id=channel_id,
                                data={"tx_hash": "0x..."},  # Would be actual hash
                                processing_time_ms=blockchain_time
                            ))

            # Complete
            events.append(PipelineEvent(
                stage=PipelineStage.COMPLETE,
                timestamp=datetime.now(),
                channel_id=channel_id,
                data={"status": "success"},
                processing_time_ms=0.0
            ))

            total_time = (time.time() - start_time) * 1000

            # Update performance tracking
            for event in events:
                if event.stage.value in self.processing_times:
                    self.processing_times[event.stage.value].append(event.processing_time_ms)

            # Add to history
            self.event_history.extend(events)

            result = PipelineResult(
                channel_id=channel_id,
                anomaly_detected=anomaly_result is not None and len(anomaly_result.anomaly_sequences) > 0,
                anomaly_score=1.0 if anomaly_result else 0.0,
                anomaly_window=anomaly_result.anomaly_sequences[0] if anomaly_result and anomaly_result.anomaly_sequences else None,
                threat_classification=threat_classification,
                explanation=explanation,
                events=events,
                total_processing_time_ms=total_time,
                success=True
            )

            logger.info(
                f"Pipeline complete for {channel_id}: "
                f"{'ANOMALY' if result.anomaly_detected else 'NORMAL'} "
                f"({total_time:.2f}ms)"
            )

            return result

        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Pipeline error for {channel_id}: {e}", exc_info=True)

            total_time = (time.time() - start_time) * 1000

            # Add error event
            events.append(PipelineEvent(
                stage=PipelineStage.COMPLETE,
                timestamp=datetime.now(),
                channel_id=channel_id,
                data={"status": "error", "error": error_message},
                processing_time_ms=0.0,
                success=False,
                error_message=error_message
            ))

            return PipelineResult(
                channel_id=channel_id,
                anomaly_detected=False,
                anomaly_score=0.0,
                events=events,
                total_processing_time_ms=total_time,
                success=False,
                error_message=error_message
            )

    async def _extract_features(
        self,
        anomaly_result: Any,
        data: np.ndarray,
        y_true: Optional[np.ndarray],
        y_hat: Optional[np.ndarray],
        error_signal: Optional[np.ndarray]
    ) -> AllFeatures:
        """Extract features from anomaly result"""
        # Get anomaly window (use first anomaly)
        if anomaly_result.anomaly_sequences:
            window_start, window_end = anomaly_result.anomaly_sequences[0]
        else:
            window_start, window_end = 0, len(data)

        # Extract features
        features = self.feature_extractor.extract_all_features(
            anomaly_result=anomaly_result,
            input_sequence=data,
            y_true=y_true or np.array([]),
            y_hat=y_hat or np.array([]),
            error_signal=error_signal or np.array([]),
            window_start=window_start,
            window_end=window_end
        )

        return features

    async def _classify_threat(
        self,
        anomaly_result: Any,
        data: np.ndarray,
        features: AllFeatures
    ) -> ThreatClassification:
        """Classify threat using threat classifier"""
        if self.threat_classifier is None:
            raise ValueError("Threat classifier not initialized")

        # Get anomaly window
        if anomaly_result.anomaly_sequences:
            window_start, window_end = anomaly_result.anomaly_sequences[0]
        else:
            window_start, window_end = 0, len(data)

        # Classify
        classification = self.threat_classifier.classify_anomaly(
            anomaly_result=anomaly_result,
            input_sequence=data,
            window_start=window_start,
            window_end=window_end
        )

        return classification

    async def _generate_explanation(
        self,
        classification: ThreatClassification,
        features: AllFeatures,
        anomaly_result: Any
    ) -> ExplainabilityResult:
        """Generate explanation using explainability engine"""
        if self.explainability_engine is None:
            raise ValueError("Explainability engine not initialized")

        # Get anomaly window
        if anomaly_result.anomaly_sequences:
            anomaly_window = anomaly_result.anomaly_sequences[0]
        else:
            anomaly_window = (0, len(features.to_array()))

        # Generate explanation
        explanation = self.explainability_engine.explain(
            classification_result=classification,
            features=features,
            anomaly_window=anomaly_window,
            error_signal=getattr(anomaly_result, 'errors_smoothed', None).tolist() if hasattr(anomaly_result, 'errors_smoothed') else None,
            detection_threshold=getattr(anomaly_result, 'threshold', None),
            channel_id="unknown"  # Would pass actual channel ID
        )

        return explanation

    def _should_trigger_alert(self, classification: ThreatClassification) -> bool:
        """Determine if alert should be triggered"""
        # Trigger alert for high-risk attacks
        if classification.threat_class == "attack" and classification.risk_score > 0.7:
            return True

        # Trigger alert for high-confidence unknown anomalies
        if classification.threat_class == "unknown" and classification.confidence > 0.8:
            return True

        return False

    async def _trigger_alert(
        self,
        channel_id: str,
        classification: ThreatClassification,
        explanation: Optional[ExplainabilityResult]
    ):
        """Trigger alert event"""
        logger.warning(
            f"ALERT: {channel_id} - {classification.threat_class} "
            f"(risk: {classification.risk_score:.2f})"
        )

        # In production, would send to alerting system (email, SMS, Slack, etc.)
        pass

    def _should_log_to_blockchain(self, classification: ThreatClassification) -> bool:
        """Determine if event should be logged to blockchain"""
        # Log only critical attacks with high confidence
        return (
            classification.threat_class == "attack" and
            classification.confidence > 0.9 and
            classification.risk_score > 0.8
        )

    async def _log_to_blockchain(
        self,
        channel_id: str,
        classification: ThreatClassification,
        explanation: Optional[ExplainabilityResult]
    ):
        """Log event to blockchain"""
        logger.info(f"Logging to blockchain: {channel_id}")

        # In production, would integrate with blockchain logger service
        # and get transaction hash
        pass

    async def _handle_alert(self, event: PipelineEvent):
        """Handle alert event"""
        logger.warning(f"Alert event: {event.channel_id}")

    async def _handle_blockchain_log(self, event: PipelineEvent):
        """Handle blockchain log event"""
        logger.info(f"Blockchain log event: {event.channel_id}")

    async def batch_process(
        self,
        channels: Dict[str, np.ndarray],
        anomaly_results: Dict[str, Any]
    ) -> Dict[str, PipelineResult]:
        """
        Process multiple channels in batch.

        Args:
            channels: Dict mapping channel_id to input data
            anomaly_results: Dict mapping channel_id to anomaly results

        Returns:
            Dict mapping channel_id to PipelineResult
        """
        results = {}

        for channel_id, data in channels.items():
            try:
                anomaly_result = anomaly_results.get(channel_id)

                result = await self.process_telemetry(
                    channel_id=channel_id,
                    data=data,
                    anomaly_detector_result=anomaly_result
                )

                results[channel_id] = result

            except Exception as e:
                logger.error(f"Error processing channel {channel_id}: {e}")

                # Add error result
                results[channel_id] = PipelineResult(
                    channel_id=channel_id,
                    anomaly_detected=False,
                    anomaly_score=0.0,
                    success=False,
                    error_message=str(e)
                )

        logger.info(f"Batch processed {len(results)} channels")
        return results

    def get_processing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get processing time statistics"""
        stats = {}

        for stage, times in self.processing_times.items():
            if times:
                stats[stage] = {
                    'mean_ms': float(np.mean(times)),
                    'std_ms': float(np.std(times)),
                    'min_ms': float(np.min(times)),
                    'max_ms': float(np.max(times)),
                    'count': len(times)
                }

        return stats

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            'stages': [stage.value for stage in PipelineStage],
            'has_threat_classifier': self.threat_classifier is not None,
            'has_explainability_engine': self.explainability_engine is not None,
            'performance_targets': {
                'anomaly_detection_ms': self.config.performance.anomaly_detection_target,
                'threat_classification_ms': self.config.performance.threat_classification_target,
                'explainability_ms': self.config.performance.explainability_target,
                'total_pipeline_ms': self.config.performance.total_pipeline_target
            },
            'event_handlers': list(self.event_handlers.keys()),
            'total_events_processed': len(self.event_history)
        }

    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")