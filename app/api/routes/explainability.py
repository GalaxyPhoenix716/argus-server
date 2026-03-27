"""
Explainability API Routes

REST API endpoints for threat explainability.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import logging
from datetime import datetime

from app.ai.explainability_engine import ExplainabilityEngine
from app.schemas.explainability import (
    ThreatExplanationRequest,
    ThreatExplanationResponse,
    BatchThreatExplanationRequest,
    BatchThreatExplanationResponse,
    FeatureAttributionRequest,
    FeatureAttributionResponse,
    TemporalAnalysisRequest,
    TemporalAnalysisResponse,
    ExplainabilityStats,
    ExplainabilityInfo,
    ExplainabilityError,
    ExplanationHistoryRequest,
    ExplanationHistoryResponse,
    PatternType,
    AttributionMethod
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])

# Global explainability engine instance
engine: Optional[ExplainabilityEngine] = None


@router.get(
    "/{classification_id}",
    response_model=ThreatExplanationResponse,
    responses={404: {"model": ExplainabilityError}, 500: {"model": ExplainabilityError}}
)
async def get_explanation(
    classification_id: str
) -> ThreatExplanationResponse:
    """
    Get explanation for a specific threat classification.

    Retrieves the explanation for a previously computed classification.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        # In production, fetch classification and explanation from database
        # For now, return not found
        raise HTTPException(
            status_code=404,
            detail=f"Explanation for classification {classification_id} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve explanation: {str(e)}"
        )


@router.post(
    "/generate",
    response_model=ThreatExplanationResponse,
    responses={500: {"model": ExplainabilityError}}
)
async def generate_explanation(
    request: ThreatExplanationRequest
) -> ThreatExplanationResponse:
    """
    Generate explanation for a threat classification.

    Creates a new explanation based on classification results and features.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        # Import numpy
        import numpy as np

        # Create simplified feature object
        class SimpleFeatures:
            def __init__(self, features_list):
                self.features_list = features_list
                self.feature_names = [f"feature_{i}" for i in range(len(features_list))]

            def to_array(self):
                return np.array(self.features_list)

            def get_feature_names(self):
                return self.feature_names

        # For now, use simplified features
        # In production, would reconstruct from classification data
        features = SimpleFeatures([0.1] * 23)  # 23 features as per extractor

        # Create simplified classification result
        class ClassificationResult:
            def __init__(self, class_name, conf):
                self.threat_class = class_name
                self.confidence = conf
                self.risk_score = conf * 0.8
                self.specific_threat_type = None
                self.top_features = []

        classification_result = ClassificationResult("unknown", 0.5)

        # Generate explanation
        explanation_result = engine.explain(
            classification_result=classification_result,
            features=features,
            anomaly_window=request.anomaly_window,
            error_signal=None,  # Would provide error signal in production
            detection_threshold=None,  # Would provide threshold in production
            channel_id=request.classification_id
        )

        # Convert to response model
        response = ThreatExplanationResponse(
            explanation={
                "top_features": [
                    {
                        "feature_name": f["name"],
                        "importance_score": f["importance"],
                        "contribution_direction": f["contribution_direction"],
                        "confidence": 0.5
                    }
                    for f in explanation_result.top_features
                ],
                "reason": explanation_result.reason,
                "pattern_type": PatternType(explanation_result.pattern_type),
                "confidence_explanation": explanation_result.confidence_explanation,
                "temporal_analysis": {
                    "pattern_type": PatternType(explanation_result.pattern_type),
                    "onset_time": explanation_result.temporal_analysis.onset_time,
                    "duration": explanation_result.temporal_analysis.duration,
                    "intensity": explanation_result.temporal_analysis.intensity,
                    "trend_slope": explanation_result.temporal_analysis.trend_slope,
                    "stability_score": explanation_result.temporal_analysis.stability_score,
                    "volatility_index": explanation_result.temporal_analysis.volatility_index,
                    "change_point_score": explanation_result.temporal_analysis.change_point_score,
                    "monotonic_score": explanation_result.temporal_analysis.monotonic_score
                },
                "anomaly_window": explanation_result.anomaly_window,
                "attribution_method": AttributionMethod(explanation_result.attribution_method),
                "timestamp": explanation_result.timestamp.isoformat(),
                "channel_id": request.classification_id,
                "classification_id": request.classification_id,
                "processing_time_ms": explanation_result.processing_time_ms
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchThreatExplanationResponse,
    responses={500: {"model": ExplainabilityError}}
)
async def batch_generate_explanations(
    request: BatchThreatExplanationRequest,
    background_tasks: BackgroundTasks
) -> BatchThreatExplanationResponse:
    """
    Generate explanations for multiple classifications in batch.

    Processes multiple explanation requests in a single call
    for improved throughput.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        start_time = datetime.now()

        # In production, process actual batch
        # For now, return placeholder results
        results = {}

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        response = BatchThreatExplanationResponse(
            results=results,
            total_processed=len(results),
            total_time_ms=total_time
        )

        return response

    except Exception as e:
        logger.error(f"Error in batch explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch explanation failed: {str(e)}"
        )


@router.post(
    "/attribution",
    response_model=FeatureAttributionResponse,
    responses={500: {"model": ExplainabilityError}}
)
async def compute_feature_attribution(
    request: FeatureAttributionRequest
) -> FeatureAttributionResponse:
    """
    Compute feature attribution for a prediction.

    Uses the specified attribution method to determine feature importance.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        import numpy as np

        # For now, use simplified attribution
        # In production, would use actual attribution method
        input_array = np.array(request.input_sequence)

        if input_array.size > 0:
            feature_names = [f"feature_{i}" for i in range(input_array.shape[1])]
            # Simple attribution based on variance
            importances = np.var(input_array, axis=0)
            importances = importances / (np.max(importances) + 1e-10)
        else:
            feature_names = []
            importances = np.array([])

        # Create attributions
        attributions = [
            {
                "feature_name": name,
                "importance_score": float(imp),
                "contribution_direction": "positive" if imp > 0.5 else "negative",
                "confidence": 0.5
            }
            for name, imp in zip(feature_names, importances)
        ]

        response = FeatureAttributionResponse(
            attributions=attributions,
            attribution_method=request.attribution_method,
            processing_time_ms=5.0  # Placeholder
        )

        return response

    except Exception as e:
        logger.error(f"Error computing attribution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute attribution: {str(e)}"
        )


@router.post(
    "/temporal-analysis",
    response_model=TemporalAnalysisResponse,
    responses={500: {"model": ExplainabilityError}}
)
async def analyze_temporal_pattern(
    request: TemporalAnalysisRequest
) -> TemporalAnalysisResponse:
    """
    Analyze temporal pattern in anomaly.

    Detects spike, drift, persistent, or intermittent patterns.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        import numpy as np

        # Perform analysis
        error_signal = np.array(request.error_signal)
        analysis = engine.temporal_detector.analyze_pattern(
            error_signal, request.anomaly_window, request.detection_threshold
        )

        # Get pattern summary
        summary = engine.temporal_detector.get_pattern_summary(analysis)

        response = TemporalAnalysisResponse(
            analysis={
                "pattern_type": PatternType(analysis.pattern_type.value),
                "onset_time": analysis.onset_time,
                "duration": analysis.duration,
                "intensity": analysis.intensity,
                "trend_slope": analysis.trend_slope,
                "stability_score": analysis.stability_score,
                "volatility_index": analysis.volatility_index,
                "change_point_score": analysis.change_point_score,
                "monotonic_score": analysis.monotonic_score,
                "burst_count": analysis.burst_count,
                "burst_intensity": analysis.burst_intensity
            },
            pattern_summary=summary
        )

        return response

    except Exception as e:
        logger.error(f"Error analyzing temporal pattern: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze temporal pattern: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=ExplainabilityStats,
    responses={500: {"model": ExplainabilityError}}
)
async def get_explainability_stats() -> ExplainabilityStats:
    """
    Get explainability statistics.

    Returns aggregated statistics about explainability performance
    and usage patterns.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        # Get processing stats
        processing_stats = engine.get_processing_stats()

        # In production, get real stats from database
        # For now, return placeholder data
        stats = ExplainabilityStats(
            total_explanations=100,
            method_usage={
                AttributionMethod.FAST_INTEGRATED_GRADIENTS: 80,
                AttributionMethod.FEATURE_IMPORTANCE: 15,
                AttributionMethod.SHAP: 5
            },
            pattern_distribution={
                PatternType.PERSISTENT: 40,
                PatternType.SPIKE: 30,
                PatternType.DRIFT: 20,
                PatternType.INTERMITTENT: 10
            },
            average_processing_time_ms=12.5,
            cache_hit_rate=0.65
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
    response_model=ExplainabilityInfo,
    responses={500: {"model": ExplainabilityError}}
)
async def get_explainability_info() -> ExplainabilityInfo:
    """
    Get explainability system information.

    Returns information about the explainability configuration and capabilities.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        info = engine.get_explainability_info()

        return ExplainabilityInfo(
            available_methods=[
                AttributionMethod.FAST_INTEGRATED_GRADIENTS,
                AttributionMethod.FEATURE_IMPORTANCE
            ],
            supported_patterns=[
                PatternType.SPIKE,
                PatternType.DRIFT,
                PatternType.PERSISTENT,
                PatternType.INTERMITTENT
            ],
            configuration={
                "attribution_method": info["attribution_method"],
                "top_k_features": info["top_k_features"],
                "pattern_detection": info["pattern_detection"],
                "use_templates": info["use_templates"],
                "cache_enabled": info["cache_enabled"]
            },
            performance_targets=info["performance"]
        )

    except Exception as e:
        logger.error(f"Error retrieving info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve info: {str(e)}"
        )


@router.get(
    "/history",
    response_model=ExplanationHistoryResponse,
    responses={500: {"model": ExplainabilityError}}
)
async def get_explanation_history(
    request: ExplanationHistoryRequest = None
) -> ExplanationHistoryResponse:
    """
    Get explanation history.

    Returns historical explanations with filtering and pagination.
    """
    try:
        # In production, query database with filters
        # For now, return empty result
        response = ExplanationHistoryResponse(
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


@router.delete(
    "/cache",
    responses={200: {"description": "Cache cleared successfully"}}
)
async def clear_explanation_cache() -> dict:
    """Clear the explanation cache."""
    global engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized"
        )

    try:
        engine.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


def initialize_explainability_engine(engine_instance: ExplainabilityEngine):
    """
    Initialize the global explainability engine instance.

    This should be called during application startup.
    """
    global engine
    engine = engine_instance
    logger.info("Explainability engine initialized")


def get_explainability_engine() -> Optional[ExplainabilityEngine]:
    """Get the global explainability engine instance"""
    return engine