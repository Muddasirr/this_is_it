#!/usr/bin/env python3
"""Re-run the backend events engine on a saved detections.json (no YOLO, no GPU).

Fast loop for event tuning: Kaggle produces detections once, then iterate here
and score with score_events.py.

  PYTHONPATH=backend python scripts/eval/reinfer_events.py \\
      data/kaggle_out/<run>/test_video_2 data/eval/reinfer/test_video_2
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.scripts.inject_detections import _load_detections  # noqa: E402
from app.services.events import infer_event_candidates  # noqa: E402
from app.services.fuse import fuse_event_candidates  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    dets = _load_detections(src / "detections.json")
    offsets: dict[str, int] = {}
    summary = src / "summary.json"
    if summary.exists():
        offsets = {s["camera"]: int(s["offset_ms"]) for s in json.loads(summary.read_text()).get("sync") or []}

    candidates = []
    for cam in sorted({d.camera_label for d in dets}):
        candidates.extend(infer_event_candidates([d for d in dets if d.camera_label == cam], cam))
    fused = fuse_event_candidates(candidates, offsets_ms=offsets)
    print("observations", dict(Counter(e.event_type for e in candidates)))
    print("canonical   ", dict(Counter(e.event_type for e in fused)))

    (out / "events.json").write_text(
        json.dumps(
            [
                {
                    "event_type": e.event_type,
                    "camera_label": e.camera_label,
                    "camera_ts_ms": e.camera_ts_ms,
                    "ts_ms": e.camera_ts_ms + offsets.get(e.camera_label, 0),
                    "confidence": round(e.confidence, 4),
                    "track_keys": e.track_keys,
                    "evidence": e.evidence,
                }
                for e in fused
            ],
            indent=2,
        )
    )
    print("wrote", out / "events.json")


if __name__ == "__main__":
    main()
