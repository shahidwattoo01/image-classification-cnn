"""Model architectures: a from-scratch CNN and ImageNet transfer-learning models.

Every model takes raw RGB images in [0, 255] and does its own input scaling, so training,
evaluation and the deployed predictor all feed images the same way.
"""
from __future__ import annotations

import keras
from keras import layers, regularizers

BACKBONES = {
    "mobilenet_v2": keras.applications.MobileNetV2,
    "resnet50_v2": keras.applications.ResNet50V2,
}


def build_custom_cnn(num_classes: int, image_size: tuple[int, int],
                     filters: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.5,
                     spatial_dropout: float = 0.1, weight_decay: float = 1e-4) -> keras.Model:
    """VGG-style CNN: [Conv-BN-ReLU] x2 + MaxPool per stage, global average pooling head.

    Regularisation: L2 weight decay on all kernels, SpatialDropout2D after each stage
    (drops whole feature maps, which suits correlated conv activations) and dropout before
    the classifier. GAP instead of Flatten keeps the head tiny (~1k params per class).
    """
    reg = regularizers.l2(weight_decay)
    inputs = keras.Input((*image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for stage, n_filters in enumerate(filters, start=1):
        for conv in (1, 2):
            x = layers.Conv2D(n_filters, 3, padding="same", use_bias=False, kernel_regularizer=reg,
                              name=f"stage{stage}_conv{conv}")(x)
            x = layers.BatchNormalization(name=f"stage{stage}_bn{conv}")(x)
            x = layers.Activation("relu", name=f"stage{stage}_relu{conv}")(x)
        x = layers.MaxPooling2D(name=f"stage{stage}_pool")(x)
        x = layers.SpatialDropout2D(spatial_dropout, name=f"stage{stage}_drop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg,
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(backbone: str, num_classes: int, image_size: tuple[int, int],
                         dropout: float = 0.3, weights: str | None = "imagenet") -> keras.Model:
    """Frozen ImageNet backbone + new classification head (phase 1 of transfer learning).

    The backbone is called with ``training=False`` so its BatchNorm layers stay in inference
    mode even after unfreezing — fine-tuning on a small dataset must not overwrite the
    ImageNet BN statistics.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}; choose from {list(BACKBONES)}")
    inputs = keras.Input((*image_size, 3), name="image")
    # MobileNetV2 and ResNet50V2 both expect inputs scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
    base = BACKBONES[backbone](include_top=False, weights=weights, input_shape=(*image_size, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=backbone)


def build_model(name: str, num_classes: int, image_size: tuple[int, int], **kwargs) -> keras.Model:
    if name == "custom_cnn":
        return build_custom_cnn(num_classes, image_size, **kwargs)
    return build_transfer_model(name, num_classes, image_size, **kwargs)


def is_transfer(name: str) -> bool:
    return name in BACKBONES


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested pretrained backbone of a transfer model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError(f"{model.name} has no nested backbone")


def unfreeze_top(model: keras.Model, n_layers: int) -> int:
    """Phase 2: unfreeze the top ``n_layers`` of the backbone, keeping BatchNorm frozen.

    Returns the number of backbone layers that are now trainable.
    """
    base = get_backbone(model)
    base.trainable = True
    cutoff = max(len(base.layers) - n_layers, 0)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    return sum(layer.trainable for layer in base.layers)


def count_params(model: keras.Model) -> dict[str, int]:
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    total = sum(int(w.numpy().size) for w in model.weights)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}
