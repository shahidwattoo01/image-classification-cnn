"""Lightweight inference wrapper used by the CLI and real-time scripts.

Loads either a Keras model (``model.keras``) or a TFLite model (``model.tflite``) plus the
``labels.json`` written at training time. Preprocessing is done with OpenCV and mirrors the
training pipeline: BGR -> RGB, bilinear resize to the training resolution, float32 in
[0, 255] (each model scales its own input).

For TFLite the heavy TensorFlow dependency is optional: the standalone ``ai-edge-litert``
or ``tflite-runtime`` interpreter is used when installed, so a deployment needs only
NumPy + OpenCV + a ~few-MB runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Uses imdecode so non-ASCII Windows paths work."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


class _KerasBackend:
    def __init__(self, path: Path):
        import keras

        self.model = keras.models.load_model(path, compile=False)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(batch, training=False))


class _TFLiteBackend:
    def __init__(self, path: Path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        # Converted with a fixed batch dimension of 1, so run images one at a time.
        outputs = []
        for image in batch:
            self.interpreter.set_tensor(self.input["index"], image[None].astype(self.input["dtype"]))
            self.interpreter.invoke()
            outputs.append(self.interpreter.get_tensor(self.output["index"])[0])
        return np.stack(outputs).astype(np.float32)


class Predictor:
    def __init__(self, model_path: str | Path, labels_path: str | Path | None = None):
        model_path = Path(model_path)
        labels_path = Path(labels_path) if labels_path else model_path.parent / "labels.json"
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        self.class_names: list[str] = meta["class_names"]
        self.image_size: tuple[int, int] = tuple(meta["image_size"])
        self.model_path = model_path
        backend = _TFLiteBackend if model_path.suffix == ".tflite" else _KerasBackend
        self._backend = backend(model_path)

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def predict_proba(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Class probabilities for a list of BGR images, shape (N, num_classes)."""
        batch = np.stack([self.preprocess(img) for img in images_bgr])
        return self._backend(batch)

    def predict(self, image: str | Path | np.ndarray, top_k: int = 3) -> list[dict]:
        """Top-k ``{"label", "confidence"}`` for one image (path or BGR array)."""
        if not isinstance(image, np.ndarray):
            image = load_image(image)
        return self.top_k(self.predict_proba([image])[0], top_k)

    def top_k(self, probs: np.ndarray, k: int = 3) -> list[dict]:
        order = np.argsort(probs)[::-1][:k]
        return [{"label": self.class_names[i], "confidence": float(probs[i])} for i in order]

    def warmup(self) -> None:
        h, w = self.image_size
        self.predict_proba([np.zeros((h, w, 3), dtype=np.uint8)])
