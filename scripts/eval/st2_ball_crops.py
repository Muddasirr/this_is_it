#!/usr/bin/env python3
"""Ball training crops from SoccerTrack v2 (HF atomscott/soccertrack-v2, CC BY 4.0) — step 1 of the
panorama ball fine-tune.

  python scripts/eval/st2_ball_crops.py plan --out data/st2_ball/tasks.tsv
  sh data/st2_ball/extract_half.sh <match> <1st|2nd> tasks.tsv <frames dir>   (in the worker: ffmpeg)
  python scripts/eval/st2_ball_crops.py cut --tasks data/st2_ball/tasks.tsv --frames <frames dir> --out data/st2_ball/crops
  python scripts/eval/st2_ball_crops.py label --tasks data/st2_ball/tasks.tsv --crops data/st2_ball/crops --out data/st2_ball/yolo

The released ball track is exact only at labelled ball actions (it is interpolated between them),
so we sample event frames only, project the ball (pitch m → image px) through the 65 annotated
pitch keypoints (≈40–100 px off on the panorama: a search prior, not a box), and cut a native-
resolution CROP×CROP window around it with a random offset (no centre prior). `label` then runs
the detectors on each crop and keeps the ball detection nearest the projected point (≤ MAX_DIST
px) as a YOLO label; crops with no confident ball near the prior are dropped.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ST2 = ROOT / "data/eval/soccertrack2"
TRAIN = ["117092", "118575", "118576", "118577", "118578", "128058"]  # dataset README train split
SIZES = {"117092": (3840, 1906)}  # the rest are 4096x1080
CROP = 960
JITTER = 300
MAX_DIST = 160


def projector(match: str):
    from scipy.interpolate import LinearNDInterpolator

    kp = json.loads((ST2 / f"raw/{match}/{match}_keypoints.json").read_text())
    pitch = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])
    img = np.array(list(kp.values()), dtype=float)
    lx, ly = LinearNDInterpolator(pitch, img[:, 0]), LinearNDInterpolator(pitch, img[:, 1])
    return lambda x, y: (float(lx(x + 52.5, y + 34.0)), float(ly(x + 52.5, y + 34.0)))


def plan(args: argparse.Namespace) -> None:
    rng = random.Random(0)
    rows = []
    for match in args.matches.split(","):
        proj = projector(match)
        w, h = SIZES.get(match, (4096, 1080))
        actions = json.loads((ST2 / f"bas/{match}/{match}_12_class_events.json").read_text())["actions"]
        for half, tag in ((1, "1st"), (2, "2nd")):
            ball = np.load(ST2 / f"ball/{match}_{tag}_ball.npz")
            row_of = {int(f): i for i, f in enumerate(ball["frame"])}
            evs = [a for a in actions if a["gameTime"].startswith(f"{half} ")]
            for a in rng.sample(evs, min(args.per_half, len(evs))):
                i = row_of.get(int(a["frame"]))
                if i is None or not np.isfinite(ball["x"][i]):
                    continue
                u, v = proj(ball["x"][i], ball["y"][i])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                x0 = int(np.clip(u - CROP / 2 + rng.uniform(-JITTER, JITTER), 0, w - CROP))
                y0 = int(np.clip(v - CROP / 2 + rng.uniform(-JITTER, JITTER), 0, h - CROP))
                rows.append((match, tag, int(a["frame"]), x0, y0, round(u - x0), round(v - y0), a["label"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    print(f"{len(rows)} crops planned → {args.out}")


def cut(args: argparse.Namespace) -> None:
    """Full event frames (data/st2_ball/extract_half.sh) → the planned CROP×CROP windows."""
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in args.tasks.read_text().splitlines():
        match, tag, frame, x0, y0, *_ = line.split("\t")
        name = f"{match}_{tag}_{int(frame):06d}"
        src = args.frames / f"{match}_{tag}" / f"{name}.jpg"
        img = cv2.imread(str(src)) if src.exists() else None
        if img is None:
            continue
        x0, y0 = int(x0), int(y0)
        cv2.imwrite(str(args.out / f"{name}.jpg"), img[y0 : y0 + CROP, x0 : x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"{n} crops → {args.out}")


def label(args: argparse.Namespace) -> None:
    import cv2
    from ultralytics import YOLO

    models = [YOLO(m) for m in args.models.split(",")]
    ball_ids = [[c for c, n in y.names.items() if n in ("sports ball", "ball", "sports_ball")] for y in models]
    out_img, out_lbl = args.out / "images", args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    stats: dict[str, int] = {}
    for line in args.tasks.read_text().splitlines():
        match, tag, frame, _x0, _y0, pu, pv, _lab = line.split("\t")
        name = f"{match}_{tag}_{int(frame):06d}"
        src = args.crops / f"{name}.jpg"
        if not src.exists():
            continue
        tried += 1
        img = cv2.imread(str(src))
        best = None
        for mi, (model, ids) in enumerate(zip(models, ball_ids)):
            r = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=ids, verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                d = float(np.hypot((x1 + x2) / 2 - float(pu), (y1 + y2) / 2 - float(pv)))
                if d <= MAX_DIST and (best is None or c > best[0]):
                    best = (c, (x1, y1, x2, y2), mi)
        if best is None:
            continue
        c, (x1, y1, x2, y2), mi = best
        stats[f"model{mi}"] = stats.get(f"model{mi}", 0) + 1
        h, w = img.shape[:2]
        (out_img / f"{name}.jpg").write_bytes(src.read_bytes())
        (out_lbl / f"{name}.txt").write_text(
            f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
        kept += 1
    print(json.dumps({"crops": tried, "labelled": kept, "by_model": stats}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--matches", default=",".join(TRAIN))
    p.add_argument("--per-half", type=int, default=400)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("cut")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("label")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", default="backend/models/ball_yolov8m.pt,backend/yolov8m.pt")
    p.add_argument("--imgsz", type=int, default=1920, help="2x upscale of the 960 crop: the ball is ~8 px")
    p.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()
    {"plan": plan, "cut": cut, "label": label}[args.cmd](args)


if __name__ == "__main__":
    main()
