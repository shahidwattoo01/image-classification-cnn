"""Evaluate trained models on the held-out test split and write a comparison report.

    python -m cnnclf.evaluate                       # every model found in artifacts/
    python -m cnnclf.evaluate --models mobilenet_v2

For each model: accuracy, top-3 accuracy, macro precision/recall/F1, per-class report,
normalised confusion matrix, parameter count, file size and single-image CPU latency.
If ``model.tflite`` exists it is evaluated too — through the OpenCV-based
:class:`~cnnclf.predictor.Predictor`, which validates the deployment preprocessing path.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, top_k_accuracy_score)

from cnnclf.config import ARTIFACTS_DIR, MODEL_NAMES, REPORTS_DIR, SPLITS_DIR
from cnnclf.data import load_splits, make_dataset
from cnnclf.predictor import Predictor, load_image
from cnnclf.utils import load_json, save_json, setup_logging

log = logging.getLogger(__name__)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict:
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    k = min(3, len(class_names))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"top{k}_accuracy": float(top_k_accuracy_score(y_true, probs, k=k, labels=labels)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.1 * len(class_names) + 1.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def measure_latency(predictor: Predictor, image: np.ndarray, runs: int = 50) -> float:
    """Median single-image latency in ms (preprocessing + forward pass, decode excluded)."""
    for _ in range(5):
        predictor.predict_proba([image])
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predictor.predict_proba([image])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_model(name: str, artifacts_dir: Path, splits_dir: Path, reports_dir: Path,
                   split: str = "test", latency_runs: int = 50) -> dict:
    import keras

    model_dir = artifacts_dir / name
    meta = load_json(model_dir / "labels.json")
    splits, class_names = load_splits(splits_dir)
    data = splits[split]
    image_size = tuple(meta["image_size"])

    model = keras.models.load_model(model_dir / "model.keras", compile=False)
    ds = make_dataset(data, len(class_names), image_size, batch_size=64, cache=False)
    probs = model.predict(ds, verbose=0)
    result = {"model": name, "split": split, "n_images": len(data),
              "params": int(model.count_params()),
              "size_mb": round((model_dir / "model.keras").stat().st_size / 1e6, 2),
              **classification_metrics(data.labels, probs, class_names)}

    sample = load_image(data.paths[0])
    result["latency_ms"] = round(measure_latency(Predictor(model_dir / "model.keras"), sample, latency_runs), 2)

    tflite_path = model_dir / "model.tflite"
    if tflite_path.exists():
        predictor = Predictor(tflite_path)
        tfl_probs = np.concatenate([predictor.predict_proba([load_image(p)]) for p in data.paths])
        tfl = classification_metrics(data.labels, tfl_probs, class_names)
        result["tflite"] = {"accuracy": tfl["accuracy"], "macro_f1": tfl["macro_f1"],
                            "size_mb": round(tflite_path.stat().st_size / 1e6, 2),
                            "latency_ms": round(measure_latency(predictor, sample, latency_runs), 2),
                            "agreement_with_keras": float((tfl_probs.argmax(1) == probs.argmax(1)).mean())}

    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, reports_dir / f"{name}_metrics.json")
    plot_confusion_matrix(result["confusion_matrix"], class_names,
                          f"{name} — {split} accuracy {result['accuracy']:.3f}",
                          reports_dir / f"{name}_confusion_matrix.png")
    return result


def write_comparison(results: list[dict], path: Path) -> str:
    rows = ["| Model | Params | Size (MB) | Accuracy | Top-3 | Macro F1 | Latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        topk = next(v for k, v in r.items() if k.startswith("top") and k.endswith("_accuracy"))
        rows.append(f"| {r['model']} | {r['params'] / 1e6:.2f}M | {r['size_mb']:.1f} | {r['accuracy']:.4f} | "
                    f"{topk:.4f} | {r['macro_f1']:.4f} | {r['latency_ms']:.1f} |")
        if "tflite" in r:
            t = r["tflite"]
            rows.append(f"| {r['model']} (TFLite) | — | {t['size_mb']:.1f} | {t['accuracy']:.4f} | — | "
                        f"{t['macro_f1']:.4f} | {t['latency_ms']:.1f} |")
    n = results[0]["n_images"] if results else 0
    text = (f"# Model comparison ({results[0]['split'] if results else 'test'} split, {n} images)\n\n"
            + "\n".join(rows)
            + "\n\nLatency = median single-image CPU inference (OpenCV preprocessing + forward pass).\n")
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Evaluate trained models on the test split.")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--latency-runs", type=int, default=50)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = p.parse_args()

    names = args.models or [m for m in MODEL_NAMES if (args.artifacts_dir / m / "model.keras").exists()]
    if not names:
        raise SystemExit("No trained models found — run `python -m cnnclf.train` first.")
    results = []
    for name in names:
        r = evaluate_model(name, args.artifacts_dir, args.splits_dir, args.reports_dir, args.split,
                           args.latency_runs)
        log.info("%-13s acc=%.4f macro-F1=%.4f latency=%.1fms", name, r["accuracy"], r["macro_f1"],
                 r["latency_ms"])
        results.append(r)
    print(write_comparison(results, args.reports_dir / "comparison.md"))


if __name__ == "__main__":
    main()
