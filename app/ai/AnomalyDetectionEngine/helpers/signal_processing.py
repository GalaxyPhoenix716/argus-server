"""
helpers/signal_processing.py
-----------------------------
Error computation and smoothing functions for the TELEMANOM pipeline.

Functions
---------
compute_errors(y_true, y_hat, config)   → smoothed error array
ewma(arr, alpha)                         → EWMA via pandas (C-backed)
compute_threshold(errors_smoothed, config, channel_id)  → float threshold
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import ChannelConfig
from .sequence_utils import count_sequences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EWMA smoothing
# ---------------------------------------------------------------------------
def ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponentially Weighted Moving Average.

    Uses pandas.Series.ewm() which is C-backed (~100× faster than a
    pure-Python loop on arrays of 8k+ elements).

    Parameters
    ----------
    arr   : 1-D float array.
    alpha : Smoothing factor in (0, 1]. Larger → more weight on recent values.

    Returns
    -------
    np.ndarray of the same length, dtype float64.
    """
    return (
        pd.Series(arr)
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .to_numpy(dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------
def compute_errors(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    config: ChannelConfig,
    channel_id: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute absolute prediction errors and apply EWMA smoothing.

    Parameters
    ----------
    y_true     : Ground-truth values.
    y_hat      : Model predictions.
    config     : ChannelConfig; `smoothing_perc` controls the EWMA span.
    channel_id : Used for log messages.

    Returns
    -------
    (errors_raw, errors_smoothed) — both shape (n,).
    """
    errors = np.abs(y_true - y_hat)
    span = max(1, int(config.smoothing_perc * len(errors)))
    alpha = 2.0 / (span + 1)
    errors_smoothed = ewma(errors, alpha)

    logger.info(
        "%s — errors: max_raw=%.4f  max_smoothed=%.4f",
        channel_id, float(np.max(errors)), float(np.max(errors_smoothed)),
    )
    return errors, errors_smoothed


# ---------------------------------------------------------------------------
# Dynamic threshold computation
# ---------------------------------------------------------------------------
def compute_threshold(
    errors_smoothed: np.ndarray,
    config: ChannelConfig,
    channel_id: str = "",
) -> float:
    """
    Non-parametric dynamic threshold (mirrors original TELEMANOM method).

    Sweeps percentile candidates and selects the threshold that maximises:
        score = mean_delta_above_threshold / number_of_anomaly_sequences

    This balances error magnitude against over-segmentation.

    Parameters
    ----------
    errors_smoothed : EWMA-smoothed error array.
    config          : ChannelConfig; `p` sets the percentile lower bound.
    channel_id      : Used for log messages.

    Returns
    -------
    threshold : float
    """
    best_threshold: Optional[float] = None
    best_score = -np.inf

    candidates = np.percentile(
        errors_smoothed, np.arange(config.p * 100, 100, 0.5)
    )

    for threshold in candidates:
        above = errors_smoothed[errors_smoothed > threshold]
        if len(above) == 0:
            continue
        mean_delta = float(np.mean(above - threshold))
        n_seqs = count_sequences(errors_smoothed > threshold)
        if n_seqs == 0:
            continue
        score = mean_delta / n_seqs
        if score > best_score:
            best_score = score
            best_threshold = threshold

    if best_threshold is None:
        best_threshold = float(np.max(errors_smoothed))

    logger.info(
        "%s — threshold: %.6f  (score: %.4f)", channel_id, best_threshold, best_score
    )
    return float(best_threshold)
