#!/usr/bin/env python3
"""Phase 0/1 detection smoke: Roboflow YOLO labels if present, else fixture + COCO yolov8n person/ball count."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "eval"
ROBO = DATA / "roboflow_futsal"
FIXTURE = DATA / "fixture_detection"


def main() -> None:
    report: dict = {"ok": True, "checks": []}

    # Harness files exist
    for rel in [
        "scripts/eval/fixtures/test_video_2_events.json",
        "scripts/eval/score_events.py",
        "scripts/eval/download_roboflow_futsal.py",
    ]:
        p = ROOT / rel
        report["checks"].append({"path": rel, "exists": p.exists()})
        if not p.exists():
            report["ok"] = False

    # Dataset or fixture
    labels = []
    if ROBO.exists():
        for split in ("train", "valid", "test", "val"):
            labels.extend((ROBO / split / "labels").glob("*.txt"))
        if not labels:
            labels = [p for p in ROBO.rglob("*.txt") if "labels" in p.parts]
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        if ROBO.exists():
            images.extend(ROBO.rglob(ext))
    images = [p for p in images if "images" in p.parts or p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if labels and images:
        report["dataset"] = {"source": "roboflow", "labels": len(labels), "images": len(images)}
        # Spot-check a non-empty label file (YOLO: cls x y w h)
        probe = next((lp for lp in labels if lp.read_text().strip()), labels[0])
        sample = [ln for ln in probe.read_text().splitlines() if ln.strip()][:8]
        bad = 0
        for line in sample:
            parts = line.split()
            if len(parts) < 5:
                bad += 1
        report["checks"].append(
            {
                "yolo_label_parse": "ok" if bad == 0 and sample else "bad",
                "bad_lines": bad,
                "probe": probe.name,
            }
        )
        if bad or not sample:
            report["ok"] = False
    else:
        FIXTURE.mkdir(parents=True, exist_ok=True)
        meta = FIXTURE / "README.txt"
        if not meta.exists():
            meta.write_text(
                "No Roboflow download yet. Run: python scripts/eval/download_roboflow_futsal.py\n"
                "Smoke still passes with fixture marker so Phase 0 can complete offline.\n"
            )
        report["dataset"] = {"source": "fixture_marker", "note": "download Roboflow for real mAP"}
        report["checks"].append({"fixture": True})

    # YOLO import + class names for sports_ball / person
    try:
        from ultralytics import YOLO

        weights = ROOT / "yolov8n.pt"
        if not weights.exists():
            weights = ROOT / "scripts" / "kaggle" / "dataset" / "yolov8n.pt"
        model = YOLO(str(weights))
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        has_person = any(v == "person" for v in names.values())
        has_ball = any(v in {"sports ball", "sports_ball"} for v in names.values())
        report["yolo"] = {"person": has_person, "sports_ball": has_ball, "weights": str(weights)}
        if not has_person:
            report["ok"] = False
        futsal = ROOT / "yolov8n_futsal.pt"
        if futsal.exists():
            fm = YOLO(str(futsal))
            fnames = fm.names if isinstance(fm.names, dict) else dict(enumerate(fm.names))
            report["futsal_yolo"] = {
                "weights": str(futsal),
                "names": list(fnames.values()),
                "has_player": any(str(v).lower() in {"player", "goalkeeper", "person"} for v in fnames.values()),
                "has_ball": any(str(v).lower() in {"ball", "sports_ball", "sports ball"} for v in fnames.values()),
            }
            if not report["futsal_yolo"]["has_player"]:
                report["ok"] = False
    except Exception as e:
        report["ok"] = False
        report["yolo_error"] = str(e)

    out = DATA / "detection_smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)
    print("GATE: detection smoke ok")


if __name__ == "__main__":
    main()
