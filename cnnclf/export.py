"""Export a trained Keras model to TensorFlow Lite for lightweight deployment.

    python -m cnnclf.export --model mobilenet_v2                    # dynamic-range int8 weights
    python -m cnnclf.export --model mobilenet_v2 --quantize float16
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import keras
import tensorflow as tf

from cnnclf.config import ARTIFACTS_DIR, MODEL_NAMES
from cnnclf.utils import setup_logging

log = logging.getLogger(__name__)


def export_tflite(model_path: Path, out_path: Path, quantize: str = "dynamic") -> Path:
    """Convert via an intermediate SavedModel (the robust path for Keras 3 models)."""
    model = keras.models.load_model(model_path, compile=False)
    tmp = Path(tempfile.mkdtemp(prefix="savedmodel_"))
    try:
        # Fix the batch dimension to 1 — the predictor feeds frames one at a time.
        input_sig = [tf.TensorSpec((1, *model.input_shape[1:]), tf.float32, name="image")]
        export_archive = keras.export.ExportArchive()
        export_archive.track(model)
        export_archive.add_endpoint("serve", lambda x: model(x, training=False), input_signature=input_sig)
        export_archive.write_out(str(tmp), verbose=False)

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
        if quantize in ("dynamic", "float16"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]
        out_path.write_bytes(converter.convert())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Export a trained model to TFLite.")
    p.add_argument("--model", choices=MODEL_NAMES, default="mobilenet_v2")
    p.add_argument("--quantize", choices=["none", "dynamic", "float16"], default="dynamic")
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = p.parse_args()

    model_dir = args.artifacts_dir / args.model
    keras_path, tflite_path = model_dir / "model.keras", model_dir / "model.tflite"
    export_tflite(keras_path, tflite_path, args.quantize)
    log.info("%s (%.1f MB) -> %s (%.1f MB, quantize=%s)", keras_path.name, keras_path.stat().st_size / 1e6,
             tflite_path.name, tflite_path.stat().st_size / 1e6, args.quantize)


if __name__ == "__main__":
    main()
