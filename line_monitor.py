"""
Line-Side Surface Defect Monitor
================================
Simulates a line-side camera feed over a directory of frames, runs each frame
through a Roboflow-trained defect detection model, maintains a rolling defect
rate, and raises an operator alert when the rate crosses a tuned threshold.

Every inference is written to an append-only audit log so any alert can be
traced back to the exact frames, detections, and confidences that caused it.

Modes:
  hosted  - Roboflow serverless inference API (fastest to demo)
  edge    - local Roboflow Inference server in Docker, no cloud dependency
            after model weights are cached (production-representative)

Usage:
  export ROBOFLOW_API_KEY=your_key_here
  python line_monitor.py --frames ./test_images --model your-workspace/your-model/1
  python line_monitor.py --frames ./test_images --model your-workspace/your-model/1 --mode edge
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from inference_sdk import InferenceHTTPClient

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HOSTED_URL = "https://serverless.roboflow.com"
EDGE_URL = "http://localhost:9001"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        sys.exit(f"No images found in {frames_dir}")
    return frames


def draw_detections(image_path: Path, predictions: list[dict], out_dir: Path) -> None:
    """Save an annotated copy of the frame for review and reporting."""
    if not HAS_CV2:
        return
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for det in predictions:
        x, y = int(det["x"]), int(det["y"])
        w, h = int(det["width"]), int(det["height"])
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / image_path.name), img)


def main() -> None:
    ap = argparse.ArgumentParser(description="Line-side surface defect monitor")
    ap.add_argument("--frames", required=True, help="Directory of frames to stream")
    ap.add_argument("--model", required=True, help="Roboflow model id, e.g. workspace/project/1")
    ap.add_argument("--mode", choices=["hosted", "edge"], default="hosted")
    ap.add_argument("--confidence", type=float, default=0.40,
                    help="Minimum confidence to count a detection (default 0.40)")
    ap.add_argument("--window", type=int, default=20,
                    help="Rolling window size in frames (default 20)")
    ap.add_argument("--alert-rate", type=float, default=0.30,
                    help="Defect rate that triggers an alert (default 0.30)")
    ap.add_argument("--cooldown", type=int, default=15,
                    help="Frames to wait after an alert before alerting again (default 15)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="Seconds between frames, simulating line speed (default 0.25)")
    ap.add_argument("--out", default="runs", help="Output directory for logs and annotated frames")
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY in the environment. Never hardcode keys.")

    api_url = HOSTED_URL if args.mode == "hosted" else EDGE_URL
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

    frames = load_frames(Path(args.frames))
    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    alerts_path = run_dir / "alerts.jsonl"
    annotated_dir = run_dir / "annotated"

    window: deque[int] = deque(maxlen=args.window)
    class_counts: Counter = Counter()
    latencies: list[float] = []
    frames_since_alert = args.cooldown  # allow an alert immediately if warranted
    alert_count = 0

    print(f"Mode: {args.mode} ({api_url})")
    print(f"Model: {args.model}")
    print(f"Window: {args.window} frames | Alert rate: {args.alert_rate:.0%} "
          f"| Confidence floor: {args.confidence}")
    print(f"Streaming {len(frames)} frames from {args.frames}\n")

    with open(audit_path, "a") as audit, open(alerts_path, "a") as alerts:
        for i, frame in enumerate(frames, 1):
            t0 = time.perf_counter()
            try:
                result = client.infer(str(frame), model_id=args.model)
            except Exception as exc:
                # An inference failure is logged, never silently dropped. On a
                # real line a failed frame is a monitoring gap, and gaps are
                # incidents, not noise.
                record = {"ts": utc_now(), "frame": frame.name, "error": str(exc)}
                audit.write(json.dumps(record) + "\n")
                print(f"[{i:>4}] {frame.name}: INFERENCE ERROR ({exc})")
                window.append(0)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            preds = [p for p in result.get("predictions", [])
                     if p.get("confidence", 0) >= args.confidence]
            defective = 1 if preds else 0
            window.append(defective)
            for p in preds:
                class_counts[p["class"]] += 1

            record = {
                "ts": utc_now(),
                "frame": frame.name,
                "mode": args.mode,
                "model": args.model,
                "latency_ms": round(latency_ms, 1),
                "detections": [
                    {"class": p["class"], "confidence": round(p["confidence"], 3),
                     "x": p["x"], "y": p["y"], "w": p["width"], "h": p["height"]}
                    for p in preds
                ],
            }
            audit.write(json.dumps(record) + "\n")

            if preds:
                draw_detections(frame, preds, annotated_dir)

            rate = sum(window) / len(window)
            frames_since_alert += 1
            status = f"defects={len(preds)} rate={rate:.0%} {latency_ms:.0f}ms"
            print(f"[{i:>4}] {frame.name}: {status}")

            if (len(window) == args.window
                    and rate >= args.alert_rate
                    and frames_since_alert >= args.cooldown):
                alert_count += 1
                frames_since_alert = 0
                alert = {
                    "ts": utc_now(),
                    "alert_id": alert_count,
                    "rolling_rate": round(rate, 3),
                    "threshold": args.alert_rate,
                    "window_frames": [f.name for f in frames[max(0, i - args.window):i]],
                    "top_classes": class_counts.most_common(3),
                }
                alerts.write(json.dumps(alert) + "\n")
                print(f"\n{'=' * 62}")
                print(f"ALERT #{alert_count}: defect rate {rate:.0%} >= "
                      f"{args.alert_rate:.0%} over last {args.window} frames")
                print(f"Top classes: {class_counts.most_common(3)}")
                print(f"Trace: {audit_path}")
                print(f"{'=' * 62}\n")

            time.sleep(args.interval)

    # Run summary: the numbers an operator review meeting actually asks for.
    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1] if len(lat_sorted) >= 20 else (lat_sorted[-1] if lat_sorted else 0)
    print("\nRun summary")
    print(f"  Frames processed : {len(frames)}")
    print(f"  Alerts raised    : {alert_count}")
    print(f"  Detections/class : {dict(class_counts)}")
    print(f"  Latency p50/p95  : {p50:.0f}ms / {p95:.0f}ms")
    print(f"  Audit log        : {audit_path}")
    print(f"  Alerts log       : {alerts_path}")
    if HAS_CV2:
        print(f"  Annotated frames : {annotated_dir}")


if __name__ == "__main__":
    main()
