#!/usr/bin/env python3
"""Dump possession timeline from detections.json (Phase 3 gate helper)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from app.services.possession import build_possession_timeline
    from app.services.vision import Detection

    raw = json.loads(args.detections.read_text())
    dets = [
        Detection(
            track_key=d["track_key"],
            camera_label=d.get("camera_label", "A"),
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            confidence=d["confidence"],
            class_name=d["class_name"],
            features=d.get("features") or {},
            bbox_series=d.get("bbox_series") or [],
        )
        for d in raw
    ]
    timeline = build_possession_timeline(dets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(timeline, indent=2))
    segs = timeline.get("segments") or []
    ball_n = timeline.get("ball_samples") or 0
    print(json.dumps({"ball_samples": ball_n, "segments": len(segs), "out": str(args.out)}, indent=2))
    if ball_n >= 80 and not segs:
        raise SystemExit("FAIL: ball ok but no possession segments")
    if ball_n >= 80:
        print("GATE: possession timeline ok")
    else:
        print("GATE: soft pass (ball_samples < 80)")


if __name__ == "__main__":
    main()
