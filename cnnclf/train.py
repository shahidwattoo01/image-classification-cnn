"""Train a classifier on the frozen splits.

    python -m cnnclf.train --model custom_cnn
    python -m cnnclf.train --model mobilenet_v2
    python -m cnnclf.train --model resnet50_v2 --epochs 8 --fine-tune-epochs 5

Transfer models train in two phases: (1) frozen backbone, new head at a normal learning
rate; (2) top backbone layers unfrozen and fine-tuned at a ~100x lower learning rate.
Outputs land in ``artifacts/<model>/``: ``model.keras`` (best val accuracy across both
phases), ``labels.json``, ``history.json``, ``training_log.csv`` and ``training_curves.png``.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import keras

from cnnclf.augment import build_augmenter
from cnnclf.config import ARTIFACTS_DIR, MODEL_DEFAULTS, MODEL_NAMES, SEED, SPLITS_DIR
from cnnclf.data import compute_class_weights, load_splits, make_dataset
from cnnclf.models import build_model, count_params, is_transfer, unfreeze_top
from cnnclf.utils import save_json, set_seed, setup_logging

log = logging.getLogger(__name__)


def compile_model(model: keras.Model, lr: float, num_classes: int, label_smoothing: float) -> None:
    k = min(3, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=k, name=f"top{k}_accuracy")],
    )


def make_callbacks(out_dir: Path, patience: int, best_so_far: float | None = None) -> list:
    return [
        keras.callbacks.ModelCheckpoint(out_dir / "model.keras", monitor="val_accuracy", mode="max",
                                        save_best_only=True, initial_value_threshold=best_so_far),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(patience // 2, 2),
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(out_dir / "training_log.csv", append=True),
    ]


def merge_history(histories: list[keras.callbacks.History]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def plot_history(history: dict[str, list[float]], path: Path, fine_tune_start: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        if fine_tune_start:
            ax.axvline(fine_tune_start + 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    defaults = MODEL_DEFAULTS[args.model]
    cfg = {k: getattr(args, k) if getattr(args, k, None) is not None else v for k, v in defaults.items()}
    image_size = (cfg["image_size"], cfg["image_size"])
    out_dir = args.output_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_log.csv").unlink(missing_ok=True)

    splits, class_names = load_splits(args.splits_dir)
    num_classes = len(class_names)
    log.info("Model %s | %d classes | train/val/test = %d/%d/%d | image %s", args.model, num_classes,
             len(splits["train"]), len(splits["val"]), len(splits["test"]), image_size)

    augmenter = None if args.augment == "none" else build_augmenter(args.augment, seed=args.seed)
    train_ds = make_dataset(splits["train"], num_classes, image_size, cfg["batch_size"], training=True,
                            augmenter=augmenter, mixup_alpha=args.mixup_alpha, seed=args.seed)
    val_ds = make_dataset(splits["val"], num_classes, image_size, cfg["batch_size"])
    class_weight = compute_class_weights(splits["train"].labels, num_classes) if args.class_weights else None

    kwargs = {"weights": None} if is_transfer(args.model) and args.no_pretrained else {}
    model = build_model(args.model, num_classes, image_size, **kwargs)
    compile_model(model, cfg["lr"], num_classes, args.label_smoothing)
    log.info("Params: %s", count_params(model))

    save_json({"model": args.model, "class_names": class_names, "image_size": list(image_size)},
              out_dir / "labels.json")

    start = time.time()
    histories = [model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"], class_weight=class_weight,
                           callbacks=make_callbacks(out_dir, args.patience), verbose=args.verbose)]
    fine_tune_start = None

    if is_transfer(args.model) and cfg.get("fine_tune_epochs", 0) > 0:
        best_phase1 = max(histories[0].history["val_accuracy"])
        fine_tune_start = len(histories[0].history["loss"])
        n_trainable = unfreeze_top(model, cfg["fine_tune_layers"])
        compile_model(model, cfg["fine_tune_lr"], num_classes, args.label_smoothing)
        log.info("Fine-tuning top %d backbone layers (%d trainable, BN frozen) at lr=%g | params: %s",
                 cfg["fine_tune_layers"], n_trainable, cfg["fine_tune_lr"], count_params(model))
        histories.append(model.fit(
            train_ds, validation_data=val_ds, initial_epoch=fine_tune_start,
            epochs=fine_tune_start + cfg["fine_tune_epochs"], class_weight=class_weight,
            callbacks=make_callbacks(out_dir, args.patience, best_so_far=best_phase1), verbose=args.verbose))

    history = merge_history(histories)
    best = keras.models.load_model(out_dir / "model.keras")
    val_loss, val_acc, val_topk = best.evaluate(val_ds, verbose=0)
    summary = {
        "model": args.model,
        "config": {**cfg, "augment": args.augment, "mixup_alpha": args.mixup_alpha,
                   "label_smoothing": args.label_smoothing, "class_weights": args.class_weights,
                   "pretrained": is_transfer(args.model) and not args.no_pretrained, "seed": args.seed},
        "epochs_run": len(history["loss"]),
        "fine_tune_start_epoch": fine_tune_start,
        "train_seconds": round(time.time() - start, 1),
        "best_val": {"loss": val_loss, "accuracy": val_acc, "top_k_accuracy": val_topk},
        "history": history,
    }
    save_json(summary, out_dir / "history.json")
    plot_history(history, out_dir / "training_curves.png", fine_tune_start)
    log.info("Done in %.0fs | best val accuracy %.4f | saved to %s", summary["train_seconds"], val_acc, out_dir)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, default="mobilenet_v2")
    p.add_argument("--epochs", type=int, help="epochs (phase 1 for transfer models)")
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--fine-tune-epochs", type=int, help="phase-2 epochs (0 disables fine-tuning)")
    p.add_argument("--fine-tune-layers", type=int, help="number of top backbone layers to unfreeze")
    p.add_argument("--fine-tune-lr", type=float)
    p.add_argument("--augment", choices=["none", "light", "medium", "strong"], default="medium")
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="0 disables MixUp")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    p.add_argument("--no-pretrained", action="store_true", help="random-init backbone (ablation/tests)")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    train(parse_args(argv))


if __name__ == "__main__":
    main()
