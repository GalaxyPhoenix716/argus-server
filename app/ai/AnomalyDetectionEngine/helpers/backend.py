"""
helpers/backend.py
------------------
Runtime detection of the inference backend (TensorFlow/Keras or ONNXRuntime).
The backend is resolved lazily once and cached for the process lifetime.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BACKEND: Optional[str] = None  # lazily initialised


def _load_backend() -> str:
    """
    Probe available packages and return the backend name.

    Priority: TensorFlow  >  ONNXRuntime.
    Raises RuntimeError if neither is installed.
    """
    try:
        import tensorflow as tf  # noqa: F401
        logger.info("Keras/TensorFlow backend loaded.")
        return "tensorflow"
    except ImportError:
        pass

    try:
        import onnxruntime  # noqa: F401
        logger.info("ONNXRuntime backend loaded.")
        return "onnxruntime"
    except ImportError:
        pass

    raise RuntimeError(
        "Neither TensorFlow nor ONNXRuntime is available.\n"
        "Install one:  pip install tensorflow\n"
        "         or:  pip install onnxruntime"
    )


def get_backend() -> str:
    """Return the current backend name, initialising if necessary."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _load_backend()
    return _BACKEND


def reset_backend() -> None:
    """Force re-detection on next call to get_backend() (useful for testing)."""
    global _BACKEND
    _BACKEND = None
