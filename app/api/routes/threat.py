"""
Threat Classification API Routes

REST API endpoints for threat classification.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import logging
from datetime import datetime

from app.ai.ThreatClassificationEngine.threat_classifier import ThreatClassifier
from app.schemas.threat import (
    ThreatClassificationRequest,
    ThreatClassificationResponse,
    BatchThreatClassificationRequest,
    BatchThreatClassificationResponse,
    ThreatClassificationStats,
    ThreatClassificationInfo,
    ThreatClassificationUpdateRequest,
    ThreatClassificationError,
    ThreatHistoryRequest,
    ThreatHistoryResponse,
    ThreatClass,
    ClassificationMethod
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/threat", tags=["threat"])

# Global classifier instance (in production, use dependency injection)
classifier: Optional[ThreatClassifier] = None


@router.post(
    "/classify",
    response_model=ThreatClassificationResponse,
    responses={500: {"model": ThreatClassificationError}}
)
async def classify_anomaly(
    request: ThreatClassificationRequest
) -> ThreatClassificationResponse:
    """
    Classify a detected anomaly.

    Takes anomaly detection results and classifies the threat type as
    attack, failure, or unknown using ensemble of ML and rule-based methods.
    """
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Threat classifier not initialized"
        )

    try:
        # Import numpy
        import numpy as np

        # Convert input sequence to numpy array
        input_sequence = np.array(request.input_sequence)
        error_signal = np.array(request.error_signal)

        # Create simplified anomaly result object
        class AnomalyResult:
            def __init__(self, window):
                self.anomaly_sequences = [window]

        anomaly_result = AnomalyResult(request.anomaly_window)

        # Perform classification
        classification = classifier.classify_anomaly(
            anomaly_result=anomaly_result,
            input_sequence=input_sequence,
            error_signal=error_signal,
            window_start=request.anomaly_window[0],
            window_end=request.anomaly_window[1],
            temporal_context=request.metadata
        )

        # Convert to response model
        response = ThreatClassificationResponse(
            threat_class=ThreatClass(classification.threat_class),
            confidence=classification.confidence,
            risk_score=classification.risk_score,
            classification_method=ClassificationMethod(classification.classification_method.value),
            specific_threat_type=classification.specific_threat_type,
            top_features=classification.top_features,
            reasoning=classification.reasoning,
            processing_time_ms=classification.processing_time_ms,
            rule_based_prediction=ThreatClass(classification.rule_based_prediction) if classification.rule_based_prediction else None,
            rule_based_confidence=classification.rule_based_confidence,
            ml_prediction=ThreatClass(classification.ml_prediction) if classification.ml_prediction else None,
            ml_confidence=classification.ml_confidence
        )

        return response

    except Exception as e:
        logger.error(f"Error classifying anomaly: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.post(
    "/classify/batch",
    response_model=BatchThreatClassificationResponse,
    responses={500: {"model": ThreatClassificationError}}
)
async def batch_classify_anomalies(
    request: BatchThreatClassificationRequest,
    background_tasks: BackgroundTasks
) -> BatchThreatClassificationResponse:
    """
    Classify multiple anomalies in batch.

    Processes multiple channel classifications in a single request
    for improved throughput.
    """
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Threat classifier not initialized"
        )

    try:
        import numpy as np

        start_time = datetime.now()

        # Prepare data
        anomaly_results = {}
        input_data = {}

        for req in request.channels:
            # Create simplified anomaly result
            class AnomalyResult:
                def __init__(self, window):
                    self.anomaly_sequences = [window]

            anomaly_result = AnomalyResult(req.anomaly_window)
            anomaly_results[req.channel_id] = anomaly_result

            # Convert input data
            input_data[req.channel_id] = np.array(req.input_sequence)

        # Perform batch classification
        results = classifier.batch_classify(anomaly_results, input_data)

        # Convert to response model
        response_results = {}
        for channel_id, classification in results.items():
            response_results[channel_id] = ThreatClassificationResponse(
                threat_class=ThreatClass(classification.threat_class),
                confidence=classification.confidence,
                risk_score=classification.risk_score,
                classification_method=ClassificationMethod(classification.classification_method.value),
                specific_threat_type=classification.specific_threat_type,
                top_features=classification.top_features,
                reasoning=classification.reasoning,
                processing_time_ms=classification.processing_time_ms,
                rule_based_prediction=ThreatClass(classification.rule_based_prediction) if classification.rule_based_prediction else None,
                rule_based_confidence=classification.rule_based_confidence,
                ml_prediction=ThreatClass(classification.ml_prediction) if classification.ml_prediction else None,
                ml_confidence=classification.ml_confidence
            )

        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        response = BatchThreatClassificationResponse(
            results=response_results,
            total_processed=len(results),
            total_time_ms=total_time
        )

        return response

    except Exception as e:
        logger.error(f"Error in batch classification: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )


@router.get(
    "/{anomaly_id}",
    response_model=ThreatClassificationResponse,
    responses={404: {"model": ThreatClassificationError}, 500: {"model": ThreatClassificationError}}
)
async def get_classification(
    anomaly_id: str
) -> ThreatClassificationResponse:
    """
    Get a specific threat classification by ID.

    Retrieves a previously computed classification result.
    """
    try:
        # In production, fetch from database
        # For now, return not found
        raise HTTPException(
            status_code=404,
            detail=f"Classification {anomaly_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving classification: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve classification: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=ThreatClassificationStats,
    responses={500: {"model": ThreatClassificationError}}
)
async def get_classification_stats() -> ThreatClassificationStats:
    """
    Get threat classification statistics.

    Returns aggregated statistics about classification performance
    and distribution.
    """
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Threat classifier not initialized"
        )

    try:
        # Get processing stats
        processing_stats = classifier.get_processing_stats()

        # In production, get real stats from database
        # For now, return placeholder data
        stats = ThreatClassificationStats(
            total_classifications=100,
            class_distribution={
                ThreatClass.ATTACK: 15,
                ThreatClass.FAILURE: 70,
                ThreatClass.UNKNOWN: 15
            },
            average_confidence=0.82,
            average_processing_time_ms=8.5,
            method_usage={
                ClassificationMethod.ENSEMBLE: 60,
                ClassificationMethod.RULE_BASED: 25,
                ClassificationMethod.ML: 15
            }
        )

        return stats

    except Exception as e:
        logger.error(f"Error retrieving stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@router.get(
    "/info",
    response_model=ThreatClassificationInfo,
    responses={500: {"model": ThreatClassificationError}}
)
async def get_classifier_info() -> ThreatClassificationInfo:
    """
    Get threat classifier information.

    Returns information about the classifier configuration and status.
    """
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Threat classifier not initialized"
        )

    try:
        info = classifier.get_classifier_info()

        return ThreatClassificationInfo(**info)

    except Exception as e:
        logger.error(f"Error retrieving info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve info: {str(e)}"
        )


@router.put(
    "/config",
    response_model=ThreatClassificationInfo,
    responses={400: {"model": ThreatClassificationError}, 500: {"model": ThreatClassificationError}}
)
async def update_classification_config(
    config: ThreatClassificationUpdateRequest
) -> ThreatClassificationInfo:
    """
    Update threat classification configuration.

    Modifies thresholds and parameters for classification decisions.
    Note: This is a simplified implementation. In production, this would
    update a persistent configuration store.
    """
    global classifier

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Threat classifier not initialized"
        )

    try:
        # Validate inputs
        if config.unknown_threshold is not None:
            if not 0.0 <= config.unknown_threshold <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="unknown_threshold must be in [0, 1]"
                )

        if config.attack_confidence_threshold is not None:
            if not 0.0 <= config.attack_confidence_threshold <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="attack_confidence_threshold must be in [0, 1]"
                )

        if config.failure_confidence_threshold is not None:
            if not 0.0 <= config.failure_confidence_threshold <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="failure_confidence_threshold must be in [0, 1]"
                )

        # In production, update actual config
        # For now, just return current info
        info = classifier.get_classifier_info()

        return ThreatClassificationInfo(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update config: {str(e)}"
        )


@router.get(
    "/history",
    response_model=ThreatHistoryResponse,
    responses={500: {"model": ThreatClassificationError}}
)
async def get_threat_history(
    request: ThreatHistoryRequest = None
) -> ThreatHistoryResponse:
    """
    Get threat classification history.

    Returns historical classification results with filtering and pagination.
    """
    try:
        # In production, query database with filters
        # For now, return empty result
        response = ThreatHistoryResponse(
            items=[],
            total=0,
            limit=request.limit if request else 100,
            offset=request.offset if request else 0,
            has_more=False
        )

        return response

    except Exception as e:
        logger.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


def initialize_classifier(classifier_instance: ThreatClassifier):
    """
    Initialize the global classifier instance.

    This should be called during application startup.
    """
    global classifier
    classifier = classifier_instance
    logger.info("Threat classifier initialized")


def get_classifier() -> Optional[ThreatClassifier]:
    """Get the global classifier instance"""
    return classifier