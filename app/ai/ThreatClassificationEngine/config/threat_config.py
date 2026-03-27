"""
Configuration for Threat Classification System

Centralized configuration management for threat classification parameters,
thresholds, and model settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Model type: 'hist_gradient_boosting', 'random_forest', 'lightgbm'
    model_type: str = "hist_gradient_boosting"

    # HistGradientBoostingClassifier parameters
    max_iter: int = 100
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10

    # Class balancing
    class_weight: Optional[str] = "balanced"  # 'balanced', 'balanced_subsample', or None
    sample_weight: Optional[str] = None  # 'focal_loss' or None

    # Model persistence
    model_format: str = "joblib"  # 'joblib' or 'onnx'
    model_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Data split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    temporal_split: bool = True  # Maintain temporal ordering

    # SMOTE for class imbalance
    use_smote: bool = True
    smote_sampling_strategy: str = "auto"
    smote_neighbors: int = 5

    # Hyperparameter tuning
    use_optuna: bool = False
    optuna_trials: int = 100
    optuna_timeout: Optional[int] = None  # Seconds

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "f1"  # 'f1', 'precision', 'recall', 'roc_auc'

    # Training parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all cores

    # Output
    save_model: bool = True
    save_path: str = "models/threat_classifier"


@dataclass
class ClassificationConfig:
    """Configuration for classification decisions"""
    # Confidence thresholds
    attack_confidence_threshold: float = 0.8
    failure_confidence_threshold: float = 0.7
    unknown_threshold: float = 0.5  # Below this → 'unknown'

    # Risk score calculation
    risk_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'confidence': 0.6,
        'anomaly_score': 0.3,
        'feature_impact': 0.1
    })

    # Fallback logic
    use_rule_based_fallback: bool = True
    rule_based_threshold: float = 0.9
    ml_fallback_threshold: float = 0.5

    # Ensemble weights (for hybrid ML + rule-based)
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'ml': 0.7,
        'rule_based': 0.3
    })


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction"""
    # Window sizes
    anomaly_window_size: int = 30
    baseline_window_size: int = 100

    # Temporal features
    rolling_window_size: int = 10
    trend_window_size: int = 20

    # PCA
    pca_components: int = 10
    pca_fit_on_normal: bool = True

    # Feature selection
    use_feature_selection: bool = True
    feature_selection_k: int = 15  # Top K features

    # Normalization
    normalize_features: bool = True
    standardization_method: str = "standard"  # 'standard', 'robust', 'minmax'


@dataclass
class RuleBasedConfig:
    """Configuration for rule-based heuristics"""
    # GPS spoofing detection
    gps_spoofing_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'gps_velocity_mismatch': 0.5,
        'position_velocity_correlation': 0.3,
        'sudden_change': 2.0
    })

    # Sensor failure detection
    sensor_failure_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'sensor_variance_threshold': 0.01,
        'stuck_value_threshold': 0.001,
        'isolation_score': 0.8
    })

    # Communication loss detection
    communication_loss_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'all_sensors_affected': 0.7,
        'persistent_pattern': 0.8,
        'correlation_breakdown': 0.6
    })

    # System drift detection
    system_drift_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'trend_r_squared': 0.7,
        'gradual_change': 0.3,
        'stability_threshold': 0.6
    })

    # Confidence calculation
    rule_confidence_base: float = 0.8
    rule_confidence_decay: float = 0.1


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability"""
    # Feature attribution
    attribution_method: str = "fast_integrated_gradients"
    attribution_steps: int = 10
    attribution_baseline: str = "mean"

    # Top features
    top_k_features: int = 5
    min_feature_importance: float = 0.1

    # Temporal analysis
    detect_patterns: bool = True
    pattern_detection_sensitivity: float = 0.5

    # Templates
    use_templates: bool = True
    template_detail_level: str = "detailed"  # 'simple', 'detailed', 'technical'

    # Performance
    cache_attributions: bool = True
    cache_size: int = 100


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # Latency targets (milliseconds)
    anomaly_detection_target: int = 50
    feature_extraction_target: int = 5
    threat_classification_target: int = 10
    explainability_target: int = 15
    total_pipeline_target: int = 100

    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds

    # Async processing
    use_async: bool = True
    max_concurrent_tasks: int = 10
    queue_maxsize: int = 1000

    # Batching
    enable_batching: bool = True
    batch_size: int = 32
    batch_timeout: float = 0.1  # seconds


@dataclass
class EdgeConfig:
    """Configuration for edge deployment"""
    # Model optimization
    quantize_model: bool = True
    quantize_precision: str = "int8"  # 'int8', 'fp16'
    prune_model: bool = False
    pruning_amount: float = 0.2

    # Model size
    max_model_size_mb: int = 10
    max_memory_usage_mb: int = 50

    # CPU optimization
    num_threads: int = 1
    use_intel_mkl: bool = False

    # Offline mode
    offline_mode: bool = False
    sync_interval: int = 300  # seconds
    local_cache_size: int = 1000


@dataclass
class ThreatClassificationConfig:
    """Main configuration class"""
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    rules: RuleBasedConfig = field(default_factory=RuleBasedConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "ThreatClassificationConfig":
        """Load configuration from environment variables"""
        return cls()

    @classmethod
    def from_file(cls, config_path: str) -> "ThreatClassificationConfig":
        """Load configuration from YAML/JSON file"""
        # Implementation would parse YAML/JSON
        # For now, return default
        return cls()

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate thresholds
        if not 0.0 <= self.classification.attack_confidence_threshold <= 1.0:
            errors.append("attack_confidence_threshold must be in [0, 1]")

        if not 0.0 <= self.classification.failure_confidence_threshold <= 1.0:
            errors.append("failure_confidence_threshold must be in [0, 1]")

        if not 0.0 <= self.classification.unknown_threshold <= 1.0:
            errors.append("unknown_threshold must be in [0, 1]")

        # Validate latency targets
        if self.performance.total_pipeline_target <= 0:
            errors.append("total_pipeline_target must be positive")

        total_expected = (
            self.performance.anomaly_detection_target +
            self.performance.feature_extraction_target +
            self.performance.threat_classification_target +
            self.performance.explainability_target
        )

        if total_expected > self.performance.total_pipeline_target:
            errors.append(
                f"Sum of individual latency targets ({total_expected}ms) exceeds "
                f"total pipeline target ({self.performance.total_pipeline_target}ms)"
            )

        # Validate edge constraints
        if self.edge.max_model_size_mb <= 0:
            errors.append("max_model_size_mb must be positive")

        if self.edge.max_memory_usage_mb <= 0:
            errors.append("max_memory_usage_mb must be positive")

        # Validate model parameters
        if self.model.max_iter <= 0:
            errors.append("max_iter must be positive")

        if not 0.0 < self.model.learning_rate <= 1.0:
            errors.append("learning_rate must be in (0, 1]")

        # Validate training configuration
        total_split = self.training.train_split + self.training.val_split + self.training.test_split
        if abs(total_split - 1.0) > 0.001:
            errors.append("train_split + val_split + test_split must equal 1.0")

        return errors

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)

    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        # Implementation would use PyYAML or similar
        # For now, just print
        print(f"Saving configuration to {config_path}")


# Default configuration instance
default_config = ThreatClassificationConfig()


def get_config() -> ThreatClassificationConfig:
    """
    Get the default threat classification configuration.

    Returns:
        ThreatClassificationConfig instance
    """
    return default_config


def create_config(**kwargs) -> ThreatClassificationConfig:
    """
    Create a configuration with custom parameters.

    Args:
        **kwargs: Custom configuration parameters

    Returns:
        ThreatClassificationConfig instance
    """
    config = ThreatClassificationConfig()

    # Update based on kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return config