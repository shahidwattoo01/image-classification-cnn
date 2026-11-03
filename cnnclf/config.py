"""Project-wide paths and defaults."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# TensorFlow "flower_photos": 3,670 photos, 5 classes — small enough that a CNN trained
# from scratch overfits quickly, which is exactly where augmentation + transfer learning pay off.
FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Per-architecture defaults. Pretrained backbones use their native ImageNet resolution;
# the from-scratch CNN uses a smaller input to keep CPU training tractable.
MODEL_DEFAULTS: dict[str, dict] = {
    "custom_cnn": {"image_size": 128, "epochs": 40, "lr": 1e-3, "batch_size": 32},
    "mobilenet_v2": {"image_size": 224, "epochs": 15, "lr": 1e-3, "batch_size": 32,
                     "fine_tune_epochs": 10, "fine_tune_layers": 40, "fine_tune_lr": 1e-5},
    "resnet50_v2": {"image_size": 224, "epochs": 10, "lr": 1e-3, "batch_size": 32,
                    "fine_tune_epochs": 8, "fine_tune_layers": 30, "fine_tune_lr": 1e-5},
}
MODEL_NAMES = tuple(MODEL_DEFAULTS)
