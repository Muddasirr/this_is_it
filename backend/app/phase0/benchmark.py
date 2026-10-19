"""CLI: run Phase 0 feasibility on a public / online match clip.

Usage:
  python -m app.phase0.benchmark /path/to/clip.mp4
  python -m app.phase0.benchmark /path/to/clip.mp4 --max-frames 200 --stride 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 pretrained detection feasibility")
    parser.add_argument("video", type=Path, help="Path to a short futsal/football clip")
    parser.add_argument("--camera", default="A", help="Camera label for track keys")
    parser.add_argument("--model", default=None, help="Ultralytics model, e.g. yolov8n.pt")
    parser.add_argument("--device", default=None, help="cpu | 0 | cuda")
    parser.add_argument("--stride", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames for quick tests")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report path")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"File not found: {args.video}", file=sys.stderr)
        return 1

    # Apply CLI overrides via env before settings cache
    import os

    if args.model:
        os.environ["YOLO_MODEL"] = args.model
    if args.device:
        os.environ["VISION_DEVICE"] = args.device
    if args.stride is not None:
        os.environ["VISION_FRAME_STRIDE"] = str(args.stride)
    if args.max_frames is not None:
        os.environ["VISION_MAX_FRAMES"] = str(args.max_frames)
    os.environ.setdefault("VISION_BACKEND", "yolo")

    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.services.yolo_backend import run_feasibility_on_video

    report = run_feasibility_on_video(str(args.video), camera_label=args.camera)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"\nWrote {args.out}", file=sys.stderr)

    go = report["go_no_go"]["suggested_continue"]
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())
