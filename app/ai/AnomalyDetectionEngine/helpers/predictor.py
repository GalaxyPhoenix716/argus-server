"""
helpers/predictor.py
--------------------
Sliding-window batch inference using the LSTM model loaded by model_loader.
Supports both TensorFlow/Keras and ONNXRuntime backends transparently.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .backend import get_backend
from .config import ChannelConfig

logger = logging.getLogger(__name__)


def predict(
    model: Any,
    data: np.ndarray,
    config: ChannelConfig,
    channel_id: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one-step-ahead predictions via a sliding window of length l_s.

    For each position i in [0, T - l_s), the model receives the window
    data[i : i + l_s] (shape: l_s × F) and produces n_predictions values.
    We keep only the first predicted value (step t+1) as y_hat[i].

    Parameters
    ----------
    model     : Keras Model or ONNXRuntime InferenceSession.
    data      : np.ndarray of shape (T, F) — full test set for the channel.
    config    : ChannelConfig with l_s, batch_size, etc.
    channel_id: Used only for log messages.

    Returns
    -------
    (y_hat, y_true)
        y_hat  : shape (T - l_s,) — model predictions.
        y_true : shape (T - l_s,) — ground-truth target (column 0).
    """
    X = data                    # (T, F)
    y_true = data[config.l_s:, 0]   # target column, starting from l_s
    n_steps = len(X) - config.l_s

    # Build all windows upfront — (n_steps, l_s, F)
    windows = np.stack(
        [X[i: i + config.l_s] for i in range(n_steps)],
        axis=0,
        dtype=np.float32,
    )

    backend = get_backend()
    preds: list[np.ndarray] = []

    if backend == "tensorflow":
        for start in range(0, n_steps, config.batch_size):
            batch = windows[start: start + config.batch_size]
            out = model.predict(batch, verbose=0)   # (B, n_predictions)
            preds.append(out[:, 0])
    else:  # onnxruntime
        input_name = model.get_inputs()[0].name
        for start in range(0, n_steps, config.batch_size):
            batch = windows[start: start + config.batch_size]
            out = model.run(None, {input_name: batch})[0]
            preds.append(out[:, 0])

    y_hat = np.concatenate(preds)
    logger.info(
        "%s — predictions generated: %d steps (backend: %s)",
        channel_id, len(y_hat), backend,
    )
    return y_hat, y_true
