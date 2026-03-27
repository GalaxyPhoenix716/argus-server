"""
ONNX Conversion Utility
========================
Converts all pre-trained Keras (.h5) LSTM models in the ARGUS anomaly
detection engine to ONNX format for backend-agnostic inference via
ONNXRuntime (no TensorFlow dependency at serving time).

Usage
-----
    # Convert all 82 channel models:
    python tools/convert_to_onnx.py

    # Convert a single channel:
    python tools/convert_to_onnx.py --channel P-1

    # Specify custom directories:
    python tools/convert_to_onnx.py --model-dir path/to/h5 --out-dir path/to/onnx

Requirements
------------
    pip install tensorflow tf2onnx onnxruntime

Notes
-----
- Input shape assumed: (batch, l_s=250, n_features)
  n_features is inferred from the model's input layer.
- All 82 models share the same LSTM architecture; only the number of input
  features varies (1 for most SMAP channels, 25 for MSL multivariate channels).
- Opset 13 is used for broad ONNXRuntime compatibility.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_DEFAULT_H5_DIR = (
    _ROOT
    / "app"
    / "ai"
    / "AnamalyDetectionEngine"
    / "data"
    / "2018-05-19_15.00.10"
    / "models"
)
_DEFAULT_ONNX_DIR = (
    _ROOT
    / "app"
    / "ai"
    / "AnamalyDetectionEngine"
    / "data"
    / "2018-05-19_15.00.10"
    / "models_onnx"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_to_onnx")


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def convert_model(h5_path: Path, onnx_path: Path, opset: int = 13) -> bool:
    """
    Convert a single .h5 Keras model to .onnx.

    Returns True on success, False on failure.
    """
    try:
        import tensorflow as tf
        import tf2onnx

        model = tf.keras.models.load_model(str(h5_path), compile=False)
        input_signature = [
            tf.TensorSpec(
                shape=[None] + list(model.input_shape[1:]),
                dtype=tf.float32,
                name="input_1",
            )
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
        )
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with protobuf
        import onnx
        onnx.save_model(onnx_model, str(onnx_path))

        logger.info("✓  %s  →  %s", h5_path.name, onnx_path.name)
        return True

    except Exception as exc:
        logger.error("✗  %s  FAILED: %s", h5_path.name, exc)
        return False


def verify_onnx_model(onnx_path: Path) -> bool:
    """Run a forward pass through the ONNX model to verify it loads correctly."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path))
        input_info = sess.get_inputs()[0]
        # Build a dummy input respecting the expected shape
        shape = input_info.shape  # e.g. [None, 250, 1] or [None, 250, 25]
        batch = 2
        l_s = shape[1] or 250
        n_features = shape[2] or 1
        dummy = np.random.randn(batch, l_s, n_features).astype(np.float32)
        out = sess.run(None, {input_info.name: dummy})
        logger.debug(
            "  Verified %s — output shape: %s", onnx_path.name, out[0].shape
        )
        return True

    except Exception as exc:
        logger.warning("  Verification failed for %s: %s", onnx_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Convert ARGUS LSTM .h5 models to ONNX format."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Single channel ID to convert (e.g. P-1). If omitted, converts all.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_H5_DIR,
        help=f"Directory containing .h5 models. Default: {_DEFAULT_H5_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_ONNX_DIR,
        help=f"Output directory for .onnx models. Default: {_DEFAULT_ONNX_DIR}",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version. Default: 13",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run a forward pass to verify each converted model (default: on).",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
    )
    args = parser.parse_args(argv)

    # Check dependencies
    missing = []
    for pkg in ("tensorflow", "tf2onnx", "onnx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(
            "Missing dependencies: %s\n"
            "Install with: pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        sys.exit(1)

    model_dir: Path = args.model_dir
    out_dir: Path = args.out_dir

    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    # Collect models to convert
    if args.channel:
        h5_files = [model_dir / f"{args.channel.upper()}.h5"]
        missing_files = [f for f in h5_files if not f.exists()]
        if missing_files:
            logger.error("Model file not found: %s", missing_files[0])
            sys.exit(1)
    else:
        h5_files = sorted(model_dir.glob("*.h5"))
        if not h5_files:
            logger.error("No .h5 files found in: %s", model_dir)
            sys.exit(1)

    logger.info("Converting %d model(s)  →  %s", len(h5_files), out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded, failed = 0, 0
    for h5_path in h5_files:
        onnx_path = out_dir / h5_path.with_suffix(".onnx").name
        ok = convert_model(h5_path, onnx_path, opset=args.opset)
        if ok:
            if args.verify:
                verify_onnx_model(onnx_path)
            succeeded += 1
        else:
            failed += 1

    logger.info(
        "\nConversion complete: %d succeeded  %d failed", succeeded, failed
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
