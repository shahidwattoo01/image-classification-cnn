import numpy as np
import pytest

from cnnclf.augment import build_augmenter, mixup
from cnnclf.data import compute_class_weights, load_splits, make_dataset, scan_image_folder
from tests.conftest import CLASSES


def test_scan_image_folder(image_folder):
    paths, labels = scan_image_folder(image_folder)
    assert len(paths) == 60
    assert sorted(set(labels)) == list(CLASSES)


def test_scan_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        scan_image_folder(tmp_path)


def test_splits_are_disjoint_stratified_and_complete(splits_dir):
    splits, class_names = load_splits(splits_dir)
    assert class_names == list(CLASSES)
    sets = [set(s.paths) for s in splits.values()]
    assert sum(len(s) for s in sets) == 60
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for split in splits.values():  # stratified: every class equally represented
        counts = np.bincount(split.labels, minlength=3)
        assert counts.max() - counts.min() <= 1


def test_load_splits_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path)


def test_class_weights_balance_inverse_frequency():
    w = compute_class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] == pytest.approx(3 * w[0])
    assert w[0] * 3 + w[1] * 1 == pytest.approx(4)


def test_dataset_shapes_and_ranges(splits_dir):
    splits, names = load_splits(splits_dir)
    ds = make_dataset(splits["val"], len(names), (32, 32), batch_size=4)
    x, y = next(iter(ds))
    assert x.shape == (4, 32, 32, 3) and x.dtype.name == "float32"
    assert 0 <= float(x.numpy().min()) and float(x.numpy().max()) <= 255
    np.testing.assert_allclose(y.numpy().sum(axis=1), 1.0)


def test_training_dataset_with_augmentation_and_mixup(splits_dir):
    splits, names = load_splits(splits_dir)
    ds = make_dataset(splits["train"], len(names), (32, 32), batch_size=8, training=True,
                      augmenter=build_augmenter("strong", seed=1), mixup_alpha=0.4)
    x, y = next(iter(ds))
    assert x.shape == (8, 32, 32, 3)
    np.testing.assert_allclose(y.numpy().sum(axis=1), 1.0, rtol=1e-5)


def test_mixup_is_convex_combination():
    import tensorflow as tf

    x = tf.constant(np.stack([np.zeros((4, 4, 3)), np.full((4, 4, 3), 255.0)]), tf.float32)
    y = tf.constant([[1.0, 0.0], [0.0, 1.0]])
    mx, my = mixup(x, y, alpha=0.4)
    # the image mixing weight must equal the label weight on class 1
    np.testing.assert_allclose(mx.numpy()[:, 0, 0, 0] / 255.0, my.numpy()[:, 1], atol=1e-5)
    assert float(tf.reduce_min(mx)) >= 0 and float(tf.reduce_max(mx)) <= 255


def test_augmenter_rejects_unknown_strength():
    with pytest.raises(ValueError):
        build_augmenter("extreme")
