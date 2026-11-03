"""Dataset download, frozen stratified splits and tf.data input pipelines.

Any dataset laid out as ``<root>/<class_name>/<image files>`` works; the TensorFlow
flower photos dataset is downloaded by default.

    python -m cnnclf.data                         # download flowers + write splits
    python -m cnnclf.data --source path/to/images # use your own image folder
"""
from __future__ import annotations

import argparse
import csv
import logging
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cnnclf.config import FLOWERS_URL, IMAGE_EXTENSIONS, PROJECT_ROOT, RAW_DIR, SEED, SPLITS_DIR
from cnnclf.utils import load_json, save_json, setup_logging

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Split:
    paths: list[str]      # absolute image paths
    labels: np.ndarray    # integer class ids, shape (N,)

    def __len__(self) -> int:
        return len(self.paths)


# --------------------------------------------------------------------------- download / scan

def download_flowers(dest: Path = RAW_DIR) -> Path:
    """Download and extract the flower photos dataset (~220 MB) once; return its root."""
    target = dest / "flower_photos"
    if target.is_dir():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "flower_photos.tgz"
    if not archive.exists():
        log.info("Downloading %s", FLOWERS_URL)
        urllib.request.urlretrieve(FLOWERS_URL, archive)
    log.info("Extracting %s", archive.name)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return target


def scan_image_folder(root: Path) -> tuple[list[Path], list[str]]:
    """Return image paths and their class names (= parent folder names)."""
    root = Path(root)
    paths, labels = [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(f)
                labels.append(class_dir.name)
    if not paths:
        raise ValueError(f"No images found under {root} (expected <root>/<class>/<image>)")
    return paths, labels


# --------------------------------------------------------------------------- splits

def make_splits(root: Path, out_dir: Path = SPLITS_DIR, val_frac: float = 0.15,
                test_frac: float = 0.15, seed: int = SEED) -> dict[str, int]:
    """Write frozen, stratified train/val/test CSVs so every model sees identical data."""
    root = Path(root).resolve()
    paths, labels = scan_image_folder(root)
    class_names = sorted(set(labels))

    train_p, hold_p, train_y, hold_y = train_test_split(
        paths, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed)
    val_p, test_p, val_y, test_y = train_test_split(
        hold_p, hold_y, test_size=test_frac / (val_frac + test_frac), stratify=hold_y,
        random_state=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, ps, ys in (("train", train_p, train_y), ("val", val_p, val_y), ("test", test_p, test_y)):
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "label"])
            writer.writerows((p.relative_to(root).as_posix(), y) for p, y in zip(ps, ys))
        sizes[name] = len(ps)

    # Store the root relative to the project when possible so committed splits stay portable.
    stored_root = root.relative_to(PROJECT_ROOT).as_posix() if root.is_relative_to(PROJECT_ROOT) else str(root)
    save_json({"root": stored_root, "class_names": class_names, "sizes": sizes,
               "class_counts": dict(sorted(Counter(labels).items())), "seed": seed},
              out_dir / "meta.json")
    return sizes


def load_splits(split_dir: Path = SPLITS_DIR) -> tuple[dict[str, Split], list[str]]:
    """Load the frozen splits written by :func:`make_splits`."""
    meta_path = Path(split_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run `python -m cnnclf.data` first.")
    meta = load_json(meta_path)
    root, class_names = PROJECT_ROOT / meta["root"], meta["class_names"]  # no-op if absolute
    index = {c: i for i, c in enumerate(class_names)}

    splits = {}
    for name in SPLIT_NAMES:
        with open(Path(split_dir) / f"{name}.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        splits[name] = Split(paths=[str(root / r["path"]) for r in rows],
                             labels=np.array([index[r["label"]] for r in rows], dtype=np.int32))
    return splits, class_names


def compute_class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    """Inverse-frequency ("balanced") class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


# --------------------------------------------------------------------------- tf.data

def decode_image(path, image_size: tuple[int, int]):
    """Read + decode + resize one image to uint8 RGB (uint8 keeps the in-memory cache small)."""
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size, method="bilinear")
    img = tf.saturate_cast(tf.round(img), tf.uint8)
    img.set_shape((*image_size, 3))
    return img


def make_dataset(split: Split, num_classes: int, image_size: tuple[int, int], batch_size: int,
                 training: bool = False, augmenter=None, mixup_alpha: float = 0.0,
                 cache: bool = True, seed: int = SEED):
    """Build a batched pipeline yielding ``(float32 images in [0, 255], one-hot labels)``.

    Order: decode -> cache -> shuffle -> batch -> augment -> mixup -> prefetch, so expensive
    decoding happens once while augmentation is re-sampled every epoch.
    """
    import tensorflow as tf

    autotune = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((split.paths, split.labels))
    ds = ds.map(lambda p, y: (decode_image(p, image_size), tf.one_hot(y, num_classes)),
                num_parallel_calls=autotune)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(split), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=autotune)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=autotune)
    if training and mixup_alpha > 0:
        from cnnclf.augment import mixup

        ds = ds.map(lambda x, y: mixup(x, y, mixup_alpha), num_parallel_calls=autotune)
    return ds.prefetch(autotune)


# --------------------------------------------------------------------------- CLI

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Prepare dataset splits.")
    parser.add_argument("--source", type=Path, default=None,
                        help="Image folder (<root>/<class>/<img>). Default: download flower photos.")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()

    root = args.source or download_flowers()
    sizes = make_splits(root, args.out_dir, args.val_frac, args.test_frac, args.seed)
    meta = load_json(args.out_dir / "meta.json")
    log.info("Classes (%d): %s", len(meta["class_names"]), meta["class_counts"])
    log.info("Split sizes: %s -> %s", sizes, args.out_dir)


if __name__ == "__main__":
    main()
