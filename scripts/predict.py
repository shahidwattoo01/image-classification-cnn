"""Classify image files or folders from the command line.

    python scripts/predict.py photo.jpg
    python scripts/predict.py samples/ --model artifacts/mobilenet_v2/model.tflite --top-k 3
    python scripts/predict.py a.jpg b.png --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnnclf.config import ARTIFACTS_DIR, IMAGE_EXTENSIONS  # noqa: E402
from cnnclf.predictor import Predictor  # noqa: E402


def collect_images(inputs: list[Path]) -> list[Path]:
    images = []
    for item in inputs:
        if item.is_dir():
            images.extend(sorted(p for p in item.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))
        elif item.exists():
            images.append(item)
        else:
            raise SystemExit(f"Not found: {item}")
    return images


def main() -> None:
    p = argparse.ArgumentParser(description="Image classification inference.")
    p.add_argument("inputs", nargs="+", type=Path, help="image files and/or folders")
    p.add_argument("--model", type=Path, default=ARTIFACTS_DIR / "mobilenet_v2" / "model.keras",
                   help="model.keras or model.tflite (labels.json is read from the same folder)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = p.parse_args()

    predictor = Predictor(args.model)
    predictor.warmup()
    results = []
    for path in collect_images(args.inputs):
        t0 = time.perf_counter()
        preds = predictor.predict(path, top_k=args.top_k)
        results.append({"image": str(path), "predictions": preds,
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        top = r["predictions"][0]
        others = ", ".join(f"{p['label']} {p['confidence']:.1%}" for p in r["predictions"][1:])
        print(f"{Path(r['image']).name:<40} {top['label']:<14} {top['confidence']:>6.1%}"
              f"   ({others})  [{r['latency_ms']} ms]")


if __name__ == "__main__":
    main()
