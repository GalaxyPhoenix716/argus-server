"""
Pydantic schemas for threat classification.

Defines request/response models for threat classification API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ThreatClass(str, Enum):
    """Threat classification types"""
    ATTACK = "attack"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class ClassificationMethod(str, Enum):
    """Classification methods"""
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"
    ML = "ml"
    UNKNOWN = "unknown"


class SpecificThreatType(str, Enum):
    """Specific threat types"""
    # Attack types
    GPS_SPOOFING = "gps_spoofing"
    SENSOR_INJECTION = "sensor_injection"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    MULTI_SENSOR_CORRUPTION = "multi_sensor_corruption"
    JAMMING = "jamming"

    # Failure types
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_DEAD = "sensor_dead"
    COMMUNICATION_LOSS = "communication_loss"
    SYSTEM_DEGRADATION = "system_degradation"
    CALIBRATION_DRIFT = "calibration_drift"


class FeatureAttribution(BaseModel):
    """Feature attribution information"""
    name: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0.0, le=1.0, description="Importance score")
    value: float = Field(..., description="Feature value")


class ThreatClassificationRequest(BaseModel):
    """Request model for threat classification"""
    channel_id: str = Field(..., description="Telemetry channel identifier")
    anomaly_window: tuple[int, int] = Field(..., description="Anomaly window (start, end)")
    input_sequence: List[List[float]] = Field(
        ...,
        description="Input sequence (timesteps, features)"
    )
    error_signal: List[float] = Field(
        ...,
        description="Smoothed error signal"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (mission phase, environment, etc.)"
    )


class ThreatClassificationResponse(BaseModel):
    """Response model for threat classification"""
    # Core classification
    threat_class: ThreatClass = Field(..., description="Threat classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")

    # Classification details
    classification_method: ClassificationMethod = Field(
        ...,
        description="Method used for classification"
    )
    specific_threat_type: Optional[SpecificThreatType] = Field(
        default=None,
        description="Specific threat type (attack/failure subtype)"
    )

    # Supporting information
    top_features: List[FeatureAttribution] = Field(
        default_factory=list,
        description="Top contributing features"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Human-readable explanation"
    )

    # Metadata
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )

    # Individual component results
    rule_based_prediction: Optional[ThreatClass] = Field(
        default=None,
        description="Rule-based classification"
    )
    rule_based_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rule-based confidence"
    )
    ml_prediction: Optional[ThreatClass] = Field(
        default=None,
        description="ML model classification"
    )
    ml_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ML model confidence"
    )


class BatchThreatClassificationRequest(BaseModel):
    """Request model for batch threat classification"""
    channels: List[ThreatClassificationRequest] = Field(
        ...,
        description="List of channel classifications to process"
    )


class BatchThreatClassificationResponse(BaseModel):
    """Response model for batch threat classification"""
    results: Dict[str, ThreatClassificationResponse] = Field(
        ...,
        description="Classification results per channel"
    )
    total_processed: int = Field(..., description="Total channels processed")
    total_time_ms: float = Field(..., description="Total processing time")


class ThreatClassificationStats(BaseModel):
    """Statistics for threat classification"""
    total_classifications: int = Field(..., description="Total classifications performed")
    class_distribution: Dict[ThreatClass, int] = Field(
        ...,
        description="Distribution of classifications"
    )
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence")
    average_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average processing time"
    )
    method_usage: Dict[ClassificationMethod, int] = Field(
        ...,
        description="Usage of different classification methods"
    )


class ThreatClassificationInfo(BaseModel):
    """Information about threat classification system"""
    fitted: bool = Field(..., description="Whether classifier is fitted")
    has_model: bool = Field(..., description="Whether ML model is available")
    model_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="ML model information"
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Classification configuration"
    )


class ThreatClassificationUpdateRequest(BaseModel):
    """Request model for updating classification configuration"""
    unknown_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold for unknown classification"
    )
    attack_confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold for attack classification"
    )
    failure_confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold for failure classification"
    )
    ensemble_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Ensemble weights for ML and rule-based methods"
    )


class ThreatClassificationError(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(default=None, description="Error details")
    error_code: Optional[str] = Field(default=None, description="Error code")


class ThreatHistoryItem(BaseModel):
    """Historical threat classification item"""
    id: str = Field(..., description="Unique identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    channel_id: str = Field(..., description="Channel identifier")
    threat_class: ThreatClass = Field(..., description="Threat classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")
    specific_threat_type: Optional[SpecificThreatType] = Field(
        default=None,
        description="Specific threat type"
    )
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")


class ThreatHistoryRequest(BaseModel):
    """Request model for threat history"""
    channel_id: Optional[str] = Field(
        default=None,
        description="Filter by channel ID"
    )
    threat_class: Optional[ThreatClass] = Field(
        default=None,
        description="Filter by threat class"
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


class ThreatHistoryResponse(BaseModel):
    """Response model for threat history"""
    items: List[ThreatHistoryItem] = Field(..., description="History items")
    total: int = Field(..., description="Total items")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")
    has_more: bool = Field(..., description="Whether more results are available")


class ModelMetadata(BaseModel):
    """Model metadata"""
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    trained_at: str = Field(..., description="Training timestamp")
    features_used: List[str] = Field(..., description="Feature names")
    n_features: int = Field(..., description="Number of features")
    n_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="Class names")
    training_samples: int = Field(..., description="Number of training samples")
    validation_score: Optional[float] = Field(
        default=None,
        description="Validation score"
    )
    hyperparameters: Dict[str, Any] = Field(
        ...,
        description="Hyperparameters"
    )