"""
helpers/__init__.py
--------------------
Public re-exports for the AnamalyDetectionEngine.helpers sub-package.
Import from here to avoid traversing sub-modules directly.
"""

from .backend import get_backend, reset_backend
from .config import ChannelConfig, _DATA_DIR, _DEFAULT_MODEL_DIR, _LABELS_CSV
from .data_loader import load_cached_yhat, load_data
from .evaluator import evaluate, evaluate_sequences
from .model_loader import load_model
from .predictor import predict
from .schemas import AnomalyResult
from .sequence_utils import (
    count_sequences,
    detect_anomalies,
    extract_sequences,
    merge_sequences,
)
from .signal_processing import compute_errors, compute_threshold, ewma

__all__ = [
    # backend
    "get_backend",
    "reset_backend",
    # config
    "ChannelConfig",
    "_DATA_DIR",
    "_DEFAULT_MODEL_DIR",
    "_LABELS_CSV",
    # data
    "load_data",
    "load_cached_yhat",
    # models
    "load_model",
    # inference
    "predict",
    # signal processing
    "ewma",
    "compute_errors",
    "compute_threshold",
    # sequences
    "count_sequences",
    "extract_sequences",
    "merge_sequences",
    "detect_anomalies",
    # evaluation
    "evaluate_sequences",
    "evaluate",
    # schemas
    "AnomalyResult",
]
