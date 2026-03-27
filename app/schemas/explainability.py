"""
Pydantic schemas for explainability.

Defines request/response models for explainability API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class PatternType(str, Enum):
    """Temporal pattern types"""
    SPIKE = "spike"
    DRIFT = "drift"
    PERSISTENT = "persistent"
    INTERMITTENT = "intermittent"


class AttributionMethod(str, Enum):
    """Feature attribution methods"""
    FAST_INTEGRATED_GRADIENTS = "fast_integrated_gradients"
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP = "shap"
    LIME = "lime"
    RULE_BASED = "rule_based"


class FeatureAttribution(BaseModel):
    """Feature attribution information"""
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Importance score")
    contribution_direction: str = Field(..., description="positive or negative")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in attribution")


class TemporalAnalysis(BaseModel):
    """Temporal pattern analysis"""
    pattern_type: PatternType = Field(..., description="Type of temporal pattern")
    onset_time: int = Field(..., description="Timestep when anomaly started")
    duration: int = Field(..., description="Duration in timesteps")
    intensity: float = Field(..., description="Peak error magnitude")
    trend_slope: float = Field(..., description="Rate of change")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Pattern stability")
    volatility_index: float = Field(..., description="Volatility measure")
    change_point_score: float = Field(..., description="Sudden change indicator")
    monotonic_score: float = Field(..., ge=0.0, le=1.0, description="Monotonic trend strength")
    burst_count: Optional[int] = Field(default=None, description="Number of bursts (intermittent)")
    burst_intensity: Optional[float] = Field(default=None, description="Average burst intensity")


class ThreatExplanation(BaseModel):
    """Complete threat explanation"""
    # Core explanation
    top_features: List[FeatureAttribution] = Field(
        ...,
        description="Top contributing features"
    )
    reason: str = Field(..., description="Natural language explanation")
    pattern_type: PatternType = Field(..., description="Temporal pattern type")
    confidence_explanation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in explanation"
    )

    # Temporal context
    temporal_analysis: TemporalAnalysis = Field(
        ...,
        description="Temporal pattern analysis"
    )
    anomaly_window: tuple[int, int] = Field(
        ...,
        description="Anomaly window (start, end)"
    )

    # Attribution details
    attribution_method: AttributionMethod = Field(
        ...,
        description="Method used for feature attribution"
    )
    attribution_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Attribution method configuration"
    )

    # Metadata
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    channel_id: str = Field(..., description="Channel identifier")
    classification_id: Optional[str] = Field(
        default=None,
        description="Associated classification ID"
    )

    # Debug information
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Debug information"
    )


class ThreatExplanationRequest(BaseModel):
    """Request model for threat explanation"""
    classification_id: str = Field(..., description="Threat classification ID")
    channel_id: str = Field(..., description="Channel identifier")
    anomaly_window: tuple[int, int] = Field(..., description="Anomaly window")
    include_raw_features: bool = Field(
        default=False,
        description="Include raw feature values"
    )
    attribution_method: AttributionMethod = Field(
        default=AttributionMethod.FAST_INTEGRATED_GRADIENTS,
        description="Attribution method to use"
    )


class ThreatExplanationResponse(BaseModel):
    """Response model for threat explanation"""
    explanation: ThreatExplanation = Field(..., description="Threat explanation")


class BatchThreatExplanationRequest(BaseModel):
    """Request model for batch explanation"""
    explanations: List[ThreatExplanationRequest] = Field(
        ...,
        description="List of explanation requests"
    )


class BatchThreatExplanationResponse(BaseModel):
    """Response model for batch explanation"""
    results: Dict[str, ThreatExplanation] = Field(
        ...,
        description="Explanation results per classification ID"
    )
    total_processed: int = Field(..., description="Total explanations generated")
    total_time_ms: float = Field(..., description="Total processing time")


class FeatureAttributionRequest(BaseModel):
    """Request model for feature attribution"""
    input_sequence: List[List[float]] = Field(
        ...,
        description="Input sequence (timesteps, features)"
    )
    model_prediction: float = Field(..., description="Model prediction")
    attribution_method: AttributionMethod = Field(
        default=AttributionMethod.FAST_INTEGRATED_GRADIENTS,
        description="Attribution method"
    )
    baseline: Optional[List[float]] = Field(
        default=None,
        description="Baseline values (defaults to feature means)"
    )
    steps: int = Field(default=10, ge=5, le=50, description="Integration steps")


class FeatureAttributionResponse(BaseModel):
    """Response model for feature attribution"""
    attributions: List[FeatureAttribution] = Field(
        ...,
        description="Feature attributions"
    )
    attribution_method: AttributionMethod = Field(
        ...,
        description="Method used"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time"
    )


class TemporalAnalysisRequest(BaseModel):
    """Request model for temporal analysis"""
    error_signal: List[float] = Field(..., description="Error signal")
    anomaly_window: tuple[int, int] = Field(..., description="Anomaly window")
    detection_threshold: float = Field(..., description="Detection threshold")
    min_duration: int = Field(default=3, ge=1, description="Minimum duration")
    sensitivity: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection sensitivity")


class TemporalAnalysisResponse(BaseModel):
    """Response model for temporal analysis"""
    analysis: TemporalAnalysis = Field(..., description="Temporal analysis result")
    pattern_summary: Dict[str, Any] = Field(
        ...,
        description="Human-readable pattern summary"
    )


class ExplainabilityStats(BaseModel):
    """Statistics for explainability system"""
    total_explanations: int = Field(..., description="Total explanations generated")
    method_usage: Dict[AttributionMethod, int] = Field(
        ...,
        description="Usage of attribution methods"
    )
    pattern_distribution: Dict[PatternType, int] = Field(
        ...,
        description="Distribution of temporal patterns"
    )
    average_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average processing time"
    )
    cache_hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cache hit rate"
    )


class ExplainabilityInfo(BaseModel):
    """Information about explainability system"""
    available_methods: List[AttributionMethod] = Field(
        ...,
        description="Available attribution methods"
    )
    supported_patterns: List[PatternType] = Field(
        ...,
        description="Supported temporal patterns"
    )
    configuration: Dict[str, Any] = Field(
        ...,
        description="Explainability configuration"
    )
    performance_targets: Dict[str, float] = Field(
        ...,
        description="Performance targets (ms)"
    )


class ExplainabilityError(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Error details")
    error_code: Optional[str] = Field(default=None, description="Error code")


class ExplanationHistoryItem(BaseModel):
    """Historical explanation item"""
    id: str = Field(..., description="Unique identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    channel_id: str = Field(..., description="Channel identifier")
    classification_id: str = Field(..., description="Associated classification ID")
    pattern_type: PatternType = Field(..., description="Temporal pattern")
    top_features: List[str] = Field(..., description="Top feature names")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")


class ExplanationHistoryRequest(BaseModel):
    """Request model for explanation history"""
    channel_id: Optional[str] = Field(
        default=None,
        description="Filter by channel ID"
    )
    pattern_type: Optional[PatternType] = Field(
        default=None,
        description="Filter by pattern type"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time (ISO-8601)"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time (ISO-8601)"
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class ExplanationHistoryResponse(BaseModel):
    """Response model for explanation history"""
    items: List[ExplanationHistoryItem] = Field(..., description="History items")
    total: int = Field(..., description="Total items")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")
    has_more: bool = Field(..., description="Whether more results are available")