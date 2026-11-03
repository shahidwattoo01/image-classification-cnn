"""Small shared helpers: logging, seeding, JSON IO."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow (via Keras) for reproducible runs."""
    import keras

    keras.utils.set_random_seed(seed)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
