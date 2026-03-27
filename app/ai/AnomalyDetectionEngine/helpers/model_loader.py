"""
helpers/model_loader.py
-----------------------
Loads pre-trained LSTM models from disk.
Supports Keras .h5 files (via TensorFlow) and .onnx files (via ONNXRuntime).
The correct loader is determined by the active backend (see backend.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .backend import get_backend
from .config import _DEFAULT_MODEL_DIR

logger = logging.getLogger(__name__)


def load_model(
    channel_id: str,
    model_dir: Optional[Path] = None,
) -> Any:
    """
    Load the inference model for a telemetry channel.

    Dispatches to the Keras or ONNX loader based on the detected backend.

    Parameters
    ----------
    channel_id : str
    model_dir : Path | None
        Directory containing <channel_id>.h5 or <channel_id>.onnx files.

    Returns
    -------
    A Keras Model (tensorflow backend) or an ONNXRuntime InferenceSession.
    """
    backend = get_backend()
    if backend == "tensorflow":
        return _load_keras_model(channel_id, model_dir)
    return _load_onnx_model(channel_id, model_dir)


def _load_keras_model(channel_id: str, model_dir: Optional[Path]) -> Any:
    """Load a Keras .h5 model without re-compiling (inference-only)."""
    import tensorflow as tf

    root = model_dir or _DEFAULT_MODEL_DIR
    model_path = root / f"{channel_id.upper()}.h5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Keras model not found: {model_path}\n"
            f"Expected a pre-trained .h5 file for channel {channel_id}."
        )
    model = tf.keras.models.load_model(str(model_path), compile=False)
    logger.info("Loaded Keras model for channel %s", channel_id)
    return model


def _load_onnx_model(channel_id: str, model_dir: Optional[Path]) -> Any:
    """
    Load an ONNXRuntime InferenceSession from a .onnx model file.

    If the .onnx file is missing, raise a helpful error directing the user
    to run tools/convert_to_onnx.py.
    """
    import onnxruntime as ort

    root = model_dir or _DEFAULT_MODEL_DIR
    model_path = root / f"{channel_id.upper()}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            f"Convert the Keras models first:\n"
            f"  python tools/convert_to_onnx.py --channel {channel_id}"
        )
    sess = ort.InferenceSession(str(model_path))
    logger.info("Loaded ONNX model for channel %s", channel_id)
    return sess
