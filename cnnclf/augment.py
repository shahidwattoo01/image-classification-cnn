"""Data augmentation (Keras preprocessing layers) and MixUp.

Augmentation runs inside the tf.data pipeline rather than inside the model, so the saved
model contains no random layers and exports cleanly to TFLite for deployment.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

AUGMENT_STRENGTHS = {
    # rotation (fraction of 2*pi), zoom, translation, contrast
    "light": (0.05, 0.10, 0.05, 0.10),
    "medium": (0.10, 0.15, 0.10, 0.20),
    "strong": (0.15, 0.25, 0.15, 0.30),
}


def build_augmenter(strength: str = "medium", seed: int | None = None) -> keras.Sequential:
    """Geometric + photometric augmentation for natural images in the [0, 255] range."""
    if strength not in AUGMENT_STRENGTHS:
        raise ValueError(f"strength must be one of {list(AUGMENT_STRENGTHS)}")
    rot, zoom, shift, contrast = AUGMENT_STRENGTHS[strength]
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(rot, fill_mode="reflect", seed=seed),
            layers.RandomZoom((-zoom, zoom), fill_mode="reflect", seed=seed),
            layers.RandomTranslation(shift, shift, fill_mode="reflect", seed=seed),
            layers.RandomContrast(contrast, seed=seed),
        ],
        name=f"augmentation_{strength}",
    )


def sample_beta(shape, alpha: float):
    """Beta(alpha, alpha) samples via two Gamma draws (tf has no native Beta sampler)."""
    g1 = tf.random.gamma(shape, alpha)
    g2 = tf.random.gamma(shape, alpha)
    return g1 / (g1 + g2)


def mixup(images, labels, alpha: float = 0.2):
    """MixUp (Zhang et al., 2018): convex combinations of image pairs and their labels."""
    batch = tf.shape(images)[0]
    lam = sample_beta([batch], alpha)
    idx = tf.random.shuffle(tf.range(batch))
    lam_x = tf.reshape(lam, [-1, 1, 1, 1])
    lam_y = tf.reshape(lam, [-1, 1])
    images = lam_x * images + (1.0 - lam_x) * tf.gather(images, idx)
    labels = lam_y * labels + (1.0 - lam_y) * tf.gather(labels, idx)
    return images, labels
