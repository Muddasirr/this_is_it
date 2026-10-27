#!/usr/bin/env python3
"""Detector A/B on broadcast football: who is a player, who is a referee (SoccerNet GSR, public).

  backend/.venv/bin/python scripts/eval/bench_gsr_roles.py --models backend/yolov8m.pt,data/eval/football_analysis_model/best.pt

Samples frames from SN-GSR-2024 (read from the remote zip, no full download) and matches each
model's boxes to the labelled ones (IoU ≥ 0.5). Per model:
  player_recall   labelled players/keepers found as a person/player/goalkeeper box
  referee_as_ref  labelled referees found *as referee* (kept out of the teams)
  referee_leak    labelled referees found as a player/person box (they would join a team)
  ball_recall     labelled balls found
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

GSR_URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2024/resolve/main/{split}.zip"
PERSON = {"person", "player", "goalkeeper"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def gt_boxes(labels: dict) -> dict[str, list[tuple[str, tuple]]]:
    cats = {c["id"]: c["name"] for c in labels["categories"]}
    files = {im["image_id"]: im["file_name"] for im in labels["images"]}
    out: dict[str, list] = defaultdict(list)
    for a in labels["annotations"]:
        b = a.get("bbox_image")
        if not b:
            continue
        role = (a.get("attributes") or {}).get("role") or cats.get(a["category_id"], "other")
        if cats.get(a["category_id"]) == "ball":
            role = "ball"
        out[files[a["image_id"]]].append((role, (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])))
    return out


CACHE = ROOT / "data/sn_gsr/cache"


def read_member(z, name: str, attempts: int = 6) -> bytes:
    """Zip member via HTTP range reads, cached on disk; rides out connection resets."""
    path = CACHE / name
    if path.exists():
        return path.read_bytes()
    for i in range(attempts):
        try:
            data = z.read(name)
            break
        except Exception:  # noqa: BLE001 — remotezip wraps resets / timeouts
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="backend/yolov8m.pt,data/eval/football_analysis_model/best.pt")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--sequences", type=int, default=6)
    ap.add_argument("--frames", type=int, default=8, help="per sequence, evenly spaced")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import cv2
    from remotezip import RemoteZip
    from ultralytics import YOLO

    from app.services.yolo_backend import _build_class_map

    models = {m: YOLO(m) for m in args.models.split(",")}
    maps = {m: _build_class_map(y) for m, y in models.items()}
    stats = {m: defaultdict(int) for m in models}
    t0 = time.time()
    with RemoteZip(GSR_URL.format(split=args.split)) as z:
        seqs = sorted({n.split("/")[0] for n in z.namelist() if n.endswith("Labels-GameState.json")})
        step = max(1, len(seqs) // args.sequences)
        for seq in seqs[::step][: args.sequences]:
            gt = gt_boxes(json.loads(read_member(z, f"{seq}/Labels-GameState.json")))
            names = sorted(gt)
            for fname in names[:: max(1, len(names) // args.frames)][: args.frames]:
                img = cv2.imdecode(np.frombuffer(read_member(z, f"{seq}/img1/{fname}"), np.uint8), cv2.IMREAD_COLOR)
                truth = gt[fname]
                for m, model in models.items():
                    r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                    dets = [(maps[m].get(int(c)), tuple(b)) for c, b in
                            zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
                    dets = [(n, b) for n, b in dets if n]
                    used: set[int] = set()
                    s = stats[m]
                    for role, tb in truth:
                        best, bi = 0.0, None
                        for j, (_, db) in enumerate(dets):
                            v = iou(tb, db)
                            if j not in used and v > best:
                                best, bi = v, j
                        hit = dets[bi][0] if bi is not None and best >= (0.2 if role == "ball" else 0.5) else None
                        if hit is not None:
                            used.add(bi)
                        key = "keeper" if role == "goalkeeper" else role
                        s[f"{key}_total"] += 1
                        if role in ("player", "goalkeeper"):
                            s["player_found"] += int(hit in PERSON)
                        elif role == "referee":
                            s["referee_as_ref"] += int(hit == "referee")
                            s["referee_leak"] += int(hit in PERSON)
                        elif role == "ball":
                            s["ball_found"] += int(hit == "sports_ball")
                    s["frames"] += 1
            print(f"{seq} done ({time.time() - t0:.0f} s)", file=sys.stderr)

    report = {}
    for m, s in stats.items():
        players = s["player_total"] + s["keeper_total"]
        report[m] = {
            "frames": s["frames"],
            "player_recall": round(s["player_found"] / max(1, players), 3),
            "referee_as_ref": round(s["referee_as_ref"] / max(1, s["referee_total"]), 3),
            "referee_leak": round(s["referee_leak"] / max(1, s["referee_total"]), 3),
            "ball_recall": round(s["ball_found"] / max(1, s["ball_total"]), 3),
            "counts": dict(s),
        }
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
