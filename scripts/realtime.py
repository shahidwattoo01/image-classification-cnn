"""Real-time classification from a webcam or video file with OpenCV.

    python scripts/realtime.py                                  # default webcam
    python scripts/realtime.py --source clip.mp4 --output annotated.mp4 --no-display
    python scripts/realtime.py --model artifacts/mobilenet_v2/model.tflite

Predictions are smoothed with an exponential moving average over frames so the label does
not flicker; a center crop matches the square training images. Press ``q`` to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnnclf.config import ARTIFACTS_DIR  # noqa: E402
from cnnclf.predictor import Predictor  # noqa: E402


def center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frame[top:top + side, left:left + side]


def draw_overlay(frame: np.ndarray, preds: list[dict], fps: float, min_conf: float) -> None:
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    cv2.rectangle(frame, (left, top), (left + side - 1, top + side - 1), (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (300, 30 + 26 * len(preds)), (0, 0, 0), -1)
    for i, p in enumerate(preds):
        label = p["label"] if i or p["confidence"] >= min_conf else "uncertain"
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, f"{label}: {p['confidence']:.0%}", (10, 26 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 if i == 0 else 0.55, color, 2 if i == 0 else 1)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time image classification.")
    p.add_argument("--model", type=Path, default=ARTIFACTS_DIR / "mobilenet_v2" / "model.keras")
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--smoothing", type=float, default=0.6, help="EMA weight on past frames (0 = off)")
    p.add_argument("--min-confidence", type=float, default=0.5, help="below this the top label shows 'uncertain'")
    p.add_argument("--output", type=Path, help="optional path to save the annotated video")
    p.add_argument("--no-display", action="store_true", help="run headless (e.g. with --output)")
    args = p.parse_args()

    predictor = Predictor(args.model)
    predictor.warmup()
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}")

    writer, ema, fps, frames = None, None, 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            probs = predictor.predict_proba([center_crop(frame)])[0]
            ema = probs if ema is None else args.smoothing * ema + (1 - args.smoothing) * probs
            dt = time.perf_counter() - t0
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            frames += 1

            draw_overlay(frame, predictor.top_k(ema, args.top_k), fps, args.min_confidence)
            if args.output is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Image classifier (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frames} frames, ~{fps:.1f} FPS model throughput")


if __name__ == "__main__":
    main()
