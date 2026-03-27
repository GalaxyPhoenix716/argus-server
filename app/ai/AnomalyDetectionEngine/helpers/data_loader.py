"""
helpers/data_loader.py
----------------------
Functions for loading raw telemetry .npy files and cached prediction
artefacts from the pre-trained 2018 TELEMANOM run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .config import _DATA_DIR, _DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)


def load_data(
    channel_id: str,
    split: str = "test",
    data_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Load the .npy telemetry array for a given channel and split.

    Parameters
    ----------
    channel_id : str
        e.g. "P-1", "A-7"
    split : str
        "train" or "test"
    data_dir : Path | None
        Root data directory (defaults to the engine data/ folder).

    Returns
    -------
    np.ndarray of shape (timesteps, features).
        Column 0 is always the target signal; the rest are exogenous features.
    """
    root = data_dir or _DATA_DIR
    path = root / split / f"{channel_id.upper()}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    logger.debug("%s [%s] loaded — shape: %s", channel_id, split, data.shape)
    return data


def load_cached_yhat(
    channel_id: str,
    run_dir: Optional[Path] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load the pre-computed y_hat (and smoothed errors if available) from the
    2018 cached TELEMANOM run — no TensorFlow / GPU required.

    Parameters
    ----------
    channel_id : str
    run_dir : Path | None
        Root of the timestamped run folder.
        Defaults to the 2018-05-19_15.00.10 directory.

    Returns
    -------
    (y_hat, errors_smoothed)
        errors_smoothed is None if the smoothed_errors file is not present.
    """
    run_path = run_dir or _DEFAULT_CACHE_DIR
    yhat_path = run_path / "y_hat" / f"{channel_id.upper()}.npy"
    es_path = run_path / "smoothed_errors" / f"{channel_id.upper()}.npy"

    if not yhat_path.exists():
        raise FileNotFoundError(f"Cached y_hat not found: {yhat_path}")

    y_hat = np.load(yhat_path)
    errors_smoothed: Optional[np.ndarray] = None

    if es_path.exists():
        errors_smoothed = np.load(es_path)
        logger.info("%s — loaded cached y_hat and smoothed errors.", channel_id)
    else:
        logger.info("%s — loaded cached y_hat only (no smoothed errors file).", channel_id)

    return y_hat, errors_smoothed
