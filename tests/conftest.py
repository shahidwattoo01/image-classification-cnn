import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CLASSES = ("circle", "square", "triangle")


def _draw(shape: str, rng: np.random.Generator, size: int = 48) -> np.ndarray:
    img = np.full((size, size, 3), rng.integers(0, 80), dtype=np.uint8)
    color = tuple(int(c) for c in rng.integers(120, 256, size=3))
    c, r = size // 2 + int(rng.integers(-4, 5)), size // 3
    if shape == "circle":
        cv2.circle(img, (c, c), r, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (c - r, c - r), (c + r, c + r), color, -1)
    else:
        pts = np.array([[c, c - r], [c - r, c + r], [c + r, c + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    return img


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory):
    """Tiny synthetic <root>/<class>/<img>.png dataset: 3 classes x 20 images."""
    root = tmp_path_factory.mktemp("shapes")
    rng = np.random.default_rng(0)
    for name in CLASSES:
        (root / name).mkdir()
        for i in range(20):
            cv2.imwrite(str(root / name / f"{i:03d}.png"), _draw(name, rng))
    return root


@pytest.fixture(scope="session")
def splits_dir(image_folder, tmp_path_factory):
    from cnnclf.data import make_splits

    out = tmp_path_factory.mktemp("splits")
    make_splits(image_folder, out, val_frac=0.2, test_frac=0.2, seed=0)
    return out
