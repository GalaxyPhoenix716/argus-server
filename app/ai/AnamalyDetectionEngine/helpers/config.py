"""
helpers/config.py
-----------------
Centralised hyperparameter defaults, filesystem paths, and the ChannelConfig
dataclass for per-channel overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Default hyperparameters (mirror original 2018 TELEMANOM run / params.log)
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "l_s": 250,             # look-back window (sequence length)
    "n_predictions": 10,    # number of steps predicted ahead per window
    "smoothing_perc": 0.05, # EWMA alpha = smoothing_perc × len(errors)
    "error_buffer": 100,    # buffer added around detected windows (timesteps)
    "p": 0.13,              # percentile lower bound for threshold search
    "window_size": 30,      # window used in threshold comparison step
    "batch_size": 512,      # inference batch size
}

# ---------------------------------------------------------------------------
# Filesystem paths (all relative to the AnamalyDetectionEngine package root)
# ---------------------------------------------------------------------------
_ENGINE_ROOT = Path(__file__).parent.parent   # …/AnamalyDetectionEngine/
_DATA_DIR = _ENGINE_ROOT / "data"
_TRAIN_DIR = _DATA_DIR / "train"
_TEST_DIR = _DATA_DIR / "test"
_LABELS_CSV = _DATA_DIR / "labeled_anomalies.csv"
_DEFAULT_MODEL_DIR = _DATA_DIR / "2018-05-19_15.00.10" / "models"
_DEFAULT_CACHE_DIR = _DATA_DIR / "2018-05-19_15.00.10"


# ---------------------------------------------------------------------------
# Per-channel hyperparameter container
# ---------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """
    Hyperparameters for a single telemetry channel.
    All fields default to the values that produced the 2018 benchmark results.
    Override per-channel if needed.
    """
    l_s: int = _DEFAULTS["l_s"]
    n_predictions: int = _DEFAULTS["n_predictions"]
    smoothing_perc: float = _DEFAULTS["smoothing_perc"]
    error_buffer: int = _DEFAULTS["error_buffer"]
    p: float = _DEFAULTS["p"]
    window_size: int = _DEFAULTS["window_size"]
    batch_size: int = _DEFAULTS["batch_size"]
