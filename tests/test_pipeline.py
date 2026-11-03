"""End-to-end: train (tiny) -> evaluate -> export TFLite -> Predictor on both backends."""
import json

import numpy as np
import pytest

from cnnclf.predictor import Predictor, load_image
from cnnclf.train import parse_args, train


@pytest.fixture(scope="module")
def trained(splits_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    summary = train(parse_args([
        "--model", "custom_cnn", "--epochs", "3", "--image-size", "32", "--batch-size", "8",
        "--splits-dir", str(splits_dir), "--output-dir", str(out), "--verbose", "0"]))
    return out / "custom_cnn", summary


def test_training_writes_artifacts(trained):
    model_dir, summary = trained
    for name in ("model.keras", "labels.json", "history.json", "training_log.csv", "training_curves.png"):
        assert (model_dir / name).exists(), name
    labels = json.loads((model_dir / "labels.json").read_text())
    assert labels["class_names"] == ["circle", "square", "triangle"]
    assert labels["image_size"] == [32, 32]
    assert summary["epochs_run"] >= 1


def test_transfer_two_phase_training(splits_dir, tmp_path):
    summary = train(parse_args([
        "--model", "mobilenet_v2", "--no-pretrained", "--epochs", "1", "--fine-tune-epochs", "1",
        "--fine-tune-layers", "10", "--image-size", "32", "--batch-size", "8",
        "--splits-dir", str(splits_dir), "--output-dir", str(tmp_path), "--verbose", "0"]))
    assert summary["fine_tune_start_epoch"] == 1
    assert summary["epochs_run"] == 2


def test_predictor_keras(trained, image_folder):
    model_dir, _ = trained
    predictor = Predictor(model_dir / "model.keras")
    preds = predictor.predict(next((image_folder / "circle").iterdir()), top_k=3)
    assert [p["label"] for p in preds] and len(preds) == 3
    confs = [p["confidence"] for p in preds]
    assert confs == sorted(confs, reverse=True)
    assert sum(confs) == pytest.approx(1.0, abs=1e-4)


def test_tflite_export_matches_keras(trained, image_folder):
    from cnnclf.export import export_tflite

    model_dir, _ = trained
    export_tflite(model_dir / "model.keras", model_dir / "model.tflite", quantize="none")
    keras_pred, tfl_pred = Predictor(model_dir / "model.keras"), Predictor(model_dir / "model.tflite")
    images = [load_image(p) for p in sorted((image_folder / "square").iterdir())[:4]]
    np.testing.assert_allclose(keras_pred.predict_proba(images), tfl_pred.predict_proba(images), atol=1e-4)


def test_evaluate_report(trained, splits_dir, tmp_path):
    from cnnclf.evaluate import evaluate_model, write_comparison

    model_dir, _ = trained
    result = evaluate_model("custom_cnn", model_dir.parent, splits_dir, tmp_path, latency_runs=3)
    assert 0 <= result["accuracy"] <= 1 and result["n_images"] == 12
    assert np.asarray(result["confusion_matrix"]).sum() == 12
    assert ("tflite" in result) == (model_dir / "model.tflite").exists()
    assert (tmp_path / "custom_cnn_confusion_matrix.png").exists()
    assert "custom_cnn" in write_comparison([result], tmp_path / "comparison.md")
