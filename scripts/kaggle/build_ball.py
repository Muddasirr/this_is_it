"""Kaggle kernel: train a dedicated ball detector and measure what it changes (roadmap W2).

Training data (YOLO format, ball class only, everything else dropped):
  tiagorodrigues26/ball-annotated-3k-yolov5-pytorch   DFL broadcast frames
  michaelmortenson/football-soccer-ball-detection-dfl DFL broadcast frames
  muddasirr/courtlens-roboflow-futsal                 futsal (Roboflow Futsal-Object-Detection_v2)
Measured:
  1. ball P / R / mAP50 on each source's held-out test split — trained model vs COCO yolov8m sports_ball
  2. DFL pass benchmark (bench_events, 400 labelled clips) with VISION_BALL_MODEL = trained weights
     — baseline F1 0.503 (EVAL.md, v11)
Output: results.zip with ball_yolov8m.pt, ball_eval.json, bench_events_ball.json (+ saved dets).

  python scripts/kaggle/build_ball.py --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle as B  # noqa: E402  (shared SETUP / GPU cells, code bundle)

KERNEL_DIR = B.ROOT / "data" / "kaggle_ball_kernel"
KERNEL_SLUG = f"{B.OWNER}/courtlens-ball-train"

TRAIN = '''
import json, os, subprocess, sys, yaml
from pathlib import Path

OUT = Path("/kaggle/working/ball"); OUT.mkdir(exist_ok=True)
DS = Path("/kaggle/working/ball_ds")

def sources():
    for y in Path("/kaggle/input").rglob("data.yaml"):
        cfg = yaml.safe_load(y.read_text())
        names = cfg["names"] if isinstance(cfg["names"], dict) else dict(enumerate(cfg["names"]))
        # "Location-of-ball", "ball_in"/"ball_out", "ball", "football": every ball class is the ball
        ball = {int(k) for k, v in names.items() if "ball" in str(v).lower() and "player" not in str(v).lower()}
        if ball:
            yield y.parent, ball

def convert(root, ball_ids, tag):
    """Symlink images, rewrite labels to class 0 = ball; returns {split: n_images}."""
    counts = {}
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        dst_i = DS / tag / split / "images"; dst_l = DS / tag / split / "labels"
        dst_i.mkdir(parents=True, exist_ok=True); dst_l.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in img_dir.iterdir():
            lab = root / split / "labels" / (img.stem + ".txt")
            rows = []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) in ball_ids:
                        rows.append(" ".join(["0", *parts[1:5]]))
            link = dst_i / img.name
            if not link.exists():
                link.symlink_to(img)
            (dst_l / (img.stem + ".txt")).write_text("\\n".join(rows))  # empty file = negative image
            n += 1
        counts[split] = n
    return counts

srcs = {}
for root, bid in sources():
    tag = "_".join(root.relative_to("/kaggle/input").parts)[-60:] or root.name
    srcs[tag] = convert(root, bid, tag)
print("sources", json.dumps(srcs, indent=1))

def lists(split):
    return [str(DS / t / split / "images") for t in srcs if (DS / t / split / "images").is_dir()]

data_yaml = DS / "ball.yaml"
data_yaml.write_text(yaml.safe_dump({"train": lists("train"), "val": lists("valid"), "names": {0: "ball"}}))

from ultralytics import YOLO
EPOCHS = int(os.environ.get("BALL_EPOCHS", "40"))
model = YOLO("yolov8m.pt")
model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=1280, batch=8, patience=12, device=0,
            project="/kaggle/working/runs", name="ball", exist_ok=True, mosaic=1.0, scale=0.5,
            close_mosaic=5, verbose=False, plots=False)
best = Path("/kaggle/working/runs/ball/weights/best.pt")
weights = OUT / "ball_yolov8m.pt"
weights.write_bytes(best.read_bytes())

# 1) per-source held-out test: trained model vs COCO yolov8m sports_ball, same metric for both
def centre_match(model, test, labels, classes=None):
    """Ball found when a prediction centre lies within the labelled box; conf 0.12 = pipeline threshold."""
    tp = fp = fn = 0
    for img in sorted(test.iterdir()):
        lab = labels / (img.stem + ".txt")
        gt = [list(map(float, l.split()[1:5])) for l in lab.read_text().splitlines() if l.strip()]
        r = model.predict(str(img), imgsz=1280, conf=0.12, classes=classes, device=0, verbose=False)[0]
        pred = r.boxes.xywhn.tolist() if r.boxes is not None else []
        used = set()
        for g in gt:
            hit = next((i for i, p in enumerate(pred) if i not in used
                        and abs(p[0] - g[0]) <= max(g[2], 0.01) and abs(p[1] - g[1]) <= max(g[3], 0.01)), None)
            if hit is None:
                fn += 1
            else:
                used.add(hit)
                tp += 1
        fp += len(pred) - len(used)
    return {"precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
            "balls": tp + fn, "false_balls": fp}

evals = {}
trained, coco = YOLO(str(weights)), YOLO("yolov8m.pt")
for tag in srcs:
    test = DS / tag / "test" / "images"
    if not test.is_dir():
        continue
    labels = DS / tag / "test" / "labels"
    one = DS / f"test_{tag}.yaml"
    one.write_text(yaml.safe_dump({"train": str(test), "val": str(test), "names": {0: "ball"}}))
    m = trained.val(data=str(one), imgsz=1280, batch=8, device=0, verbose=False, plots=False)
    evals[tag] = {
        "trained_map50": round(float(m.box.map50), 3),
        "trained": centre_match(trained, test, labels),
        "coco_yolov8m_sports_ball": centre_match(coco, test, labels, classes=[32]),
    }
(OUT / "ball_eval.json").write_text(json.dumps({"sources": srcs, "epochs": EPOCHS, "test": evals}, indent=2))
print(json.dumps(evals, indent=1))

# 2) DFL pass benchmark with the trained ball model (baseline F1 0.503)
env = dict(os.environ, PYTHONPATH=str(BACKEND), VISION_DEVICE="0", VISION_TARGET_FPS="10", VISION_IMGSZ="0",
           VISION_JERSEY_OCR="0", VISION_TEAM_EMBED="0", VISION_REID="1", YOLO_MODEL="yolov8m.pt",
           VISION_BALL_MODEL=str(weights))
r = subprocess.run([sys.executable, "-m", "app.scripts.bench_events", "--clips", "/kaggle/input",
                    "--out", str(OUT / "bench_events_ball.json"), "--per-class", "150",
                    "--save-dets", str(OUT / "bench_dets_ball")], cwd=BACKEND, env=env, capture_output=True, text=True)
(OUT / "bench_events_ball.log").write_text(r.stdout + "\\n" + r.stderr)
print(r.stdout[-2500:], r.stderr[-1500:])

import zipfile
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(OUT.parent))
'''


def build() -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            B._cell("# CourtLens ball detector training (generated by scripts/kaggle/build_ball.py)", "markdown"),
            B._cell(B.SETUP.replace("__BUILD_ID__", B.BUILD_ID)),
            B._cell(B.GPU),
            B._cell(TRAIN),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = KERNEL_DIR / "courtlens_ball_train.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_SLUG,
        "title": "courtlens-ball-train",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            B.CODE_SLUG,
            "tiagorodrigues26/ball-annotated-3k-yolov5-pytorch",
            "michaelmortenson/football-soccer-ball-detection-dfl",
            f"{B.OWNER}/courtlens-roboflow-futsal",
            "alejopaullier/dfl-clips",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    return path


def push() -> None:
    """Upload the current backend (new BUILD_ID) and wait until Kaggle has processed it."""
    B.build_code_zip()
    subprocess.run(["kaggle", "datasets", "version", "-p", str(B.CODE_DIR), "-m", "ball model", "-r", "skip"], check=True)
    for _ in range(60):
        time.sleep(20)
        st = subprocess.run(["kaggle", "datasets", "status", B.CODE_SLUG], capture_output=True, text=True).stdout
        if "ready" in st.lower():
            break
    time.sleep(30)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    print("kernel →", build())
    if args.push:
        push()


if __name__ == "__main__":
    main()
