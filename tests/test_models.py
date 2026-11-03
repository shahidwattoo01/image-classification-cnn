import numpy as np
import pytest
from keras import layers

from cnnclf.models import build_model, count_params, get_backbone, unfreeze_top

SIZE = (32, 32)


def test_custom_cnn_outputs_probabilities():
    model = build_model("custom_cnn", 4, SIZE)
    probs = np.asarray(model(np.random.uniform(0, 255, (2, *SIZE, 3)).astype("float32"), training=False))
    assert probs.shape == (2, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_custom_cnn_is_regularised():
    model = build_model("custom_cnn", 4, SIZE)
    assert model.losses, "expected L2 regularisation losses"
    assert any(isinstance(layer, layers.SpatialDropout2D) for layer in model.layers)


@pytest.mark.parametrize("backbone", ["mobilenet_v2", "resnet50_v2"])
def test_transfer_model_backbone_frozen(backbone):
    model = build_model(backbone, 5, SIZE, weights=None)
    assert not get_backbone(model).trainable
    params = count_params(model)
    head = model.get_layer("predictions")
    head_params = int(np.prod(head.kernel.shape)) + head.units
    assert params["trainable"] == head_params


def test_unfreeze_top_keeps_batchnorm_frozen():
    model = build_model("mobilenet_v2", 3, SIZE, weights=None)
    n = unfreeze_top(model, 20)
    base = get_backbone(model)
    assert 0 < n <= 20
    assert all(not l.trainable for l in base.layers[:-20])
    assert all(not l.trainable for l in base.layers if isinstance(l, layers.BatchNormalization))
    assert count_params(model)["trainable"] > 3 * (base.output.shape[-1] + 1)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("vgg_nope", 3, SIZE)
