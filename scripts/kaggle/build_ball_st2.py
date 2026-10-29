"""Kaggle kernel: fine-tune the ball detector for the fixed panoramic camera (SoccerTrack v2).

Input dataset muddasirr/courtlens-st2-ball (scripts/eval/st2_ball_crops.py + data/st2_ball/extract_half.py):
  crops/<match>_<half>_<frame>.jpg   960x960 native-resolution windows at labelled ball actions
  tasks.tsv / tasks_m7.tsv           per crop: projected ball (px in the crop) — ≈40–100 px off
Kernel:
  1. auto-label the train crops: detection nearest the projected ball (≤ 220 px, nearest), from the current
     ball model or COCO yolov8m at 2x upscale (the ball is ~8 px); a contact sheet is saved to check
  2. fine-tune ball_yolov8m.pt on them + the original DFL / futsal ball sets (no forgetting)
  3. held-out M7 (test match, never trained on): share of crops with a ball found near the projected
     ball ("found") and balls predicted far from it ("far", ≈ false balls), old vs new, at the
     pipeline's native scale (imgsz 960 on a 960 crop) and 2x; plus the old DFL test splits.
Output: results.zip with ball_st2.pt, st2_eval.json, label_sheet.jpg.

  python scripts/kaggle/build_ball_st2.py --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "muddasirr"
KERNEL_DIR = ROOT / "data" / "kaggle_ball_st2_kernel"
KERNEL_SLUG = f"{OWNER}/courtlens-ball-st2"
DATA_DIR = ROOT / "data" / "st2_ball" / "kaggle_dataset"
DATA_SLUG = f"{OWNER}/courtlens-st2-ball"

CODE = r'''
import json, os, random, subprocess, sys, zipfile
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.40"], check=True)
import cv2, numpy as np, yaml
from ultralytics import YOLO, settings
settings.update({"raytune": False, "wandb": False, "mlflow": False, "comet": False, "clearml": False, "dvc": False, "tensorboard": False})

INP = Path("/kaggle/input")
ST2 = next(p.parent for p in INP.rglob("tasks.tsv"))
CROPS = ST2 / "crops"
OLD = str(next(INP.rglob("ball_yolov8m.pt")))
OUT = Path("/kaggle/working/out"); OUT.mkdir(exist_ok=True)
DS = Path("/kaggle/working/ds")
MAX_DIST, LABEL_CONF = 220, 0.10  # projection is up to ~170 px off; spare balls lie around → nearest wins

def rows(name):
    seen = set()  # two events can share a frame → one crop
    for line in (ST2 / name).read_text().splitlines():
        m, tag, f, x0, y0, pu, pv, lab = line.split("\t")
        p = CROPS / f"{m}_{tag}_{int(f):06d}.jpg"
        if p.exists() and p not in seen:
            seen.add(p)
            yield p, float(pu), float(pv), m, int(x0), int(y0)

def balls(model, img, imgsz, conf, classes):
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=classes, device=0, verbose=False)[0]
    return [((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2, c)
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

old, coco = YOLO(OLD), YOLO("yolov8m.pt")
LABELLERS = [(old, [0]), (coco, [32])]

# 1. auto-label (train matches only); hold out M8 (128058) as the validation split.
# Filters (label sheet v1: ~10 % wrong): off the pitch (floodlights at night → outside the
# keypoint hull + 40 px) and static objects (same image spot ≥ 3 times in a match: spare balls, cones).
hulls = {}
for kp in ST2.glob("keypoints/*.json"):
    hulls[kp.stem] = cv2.convexHull(np.array(list(json.loads(kp.read_text()).values()), np.float32))
train_rows = list(rows("tasks.tsv"))
found = []
for p, pu, pv, m, x0, y0 in train_rows:
    img = cv2.imread(str(p))
    best = None
    for model, cls in LABELLERS:
        for cx, cy, x1, y1, x2, y2, c in balls(model, img, 1920, LABEL_CONF, cls):
            d = np.hypot(cx - pu, cy - pv)
            if d <= MAX_DIST and (best is None or d < best[-1]):
                best = (x1, y1, x2, y2, d)
    if best is not None:
        found.append((p, m, x0, y0, img.shape, best))
spot = lambda m, x0, y0, b: (m, round((x0 + (b[0] + b[2]) / 2) / 10), round((y0 + (b[1] + b[3]) / 2) / 10))
from collections import Counter
spots = Counter(spot(m, x0, y0, b) for _, m, x0, y0, _, b in found)
kept = {"train": 0, "valid": 0, "tried": len(train_rows), "found": len(found), "off_pitch": 0, "static": 0}
sheet = []
for split in ("train", "valid"):
    (DS / "st2" / split / "images").mkdir(parents=True, exist_ok=True)
    (DS / "st2" / split / "labels").mkdir(parents=True, exist_ok=True)
for p, m, x0, y0, (h, w, _), b in found:
    x1, y1, x2, y2, _ = b
    if m in hulls and cv2.pointPolygonTest(hulls[m], (x0 + (x1 + x2) / 2, y0 + (y1 + y2) / 2), True) < -40:
        kept["off_pitch"] += 1
        continue
    if spots[spot(m, x0, y0, b)] >= 3:
        kept["static"] += 1
        continue
    split = "valid" if m == "128058" else "train"
    (DS / "st2" / split / "images" / p.name).symlink_to(p)
    (DS / "st2" / split / "labels" / f"{p.stem}.txt").write_text(
        f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
    kept[split] += 1
    if len(sheet) < 48 and random.random() < 0.05:
        img = cv2.imread(str(p))
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        t = img[max(0, cy - 60):cy + 60, max(0, cx - 80):cx + 80].copy()
        t = cv2.resize(cv2.copyMakeBorder(t, 0, 120 - t.shape[0], 0, 160 - t.shape[1], cv2.BORDER_CONSTANT), (320, 240))
        cv2.rectangle(t, (int((x1 - max(0, cx - 80)) * 2), int((y1 - max(0, cy - 60)) * 2)),
                      (int((x2 - max(0, cx - 80)) * 2), int((y2 - max(0, cy - 60)) * 2)), (0, 0, 255), 1)
        sheet.append(t)
print("labels", kept, flush=True)
if sheet:
    sheet += [np.zeros_like(sheet[0])] * (-len(sheet) % 8)
    cv2.imwrite(str(OUT / "label_sheet.jpg"), np.vstack([np.hstack(sheet[i:i + 8]) for i in range(0, len(sheet), 8)]))

# old ball sets (DFL broadcast + futsal), class → 0, so the panorama is learned without forgetting
def old_sets():
    for y in INP.rglob("data.yaml"):
        if "st2" in str(y):
            continue
        cfg = yaml.safe_load(y.read_text())
        names = cfg["names"] if isinstance(cfg["names"], dict) else dict(enumerate(cfg["names"]))
        ids = {int(k) for k, v in names.items() if "ball" in str(v).lower() and "player" not in str(v).lower()}
        if ids:
            yield y.parent, ids
extra = {}
for root, ids in old_sets():
    tag = "_".join(root.relative_to(INP).parts)[-50:]
    for split in ("train", "valid", "test"):
        src = root / split / "images"
        if not src.is_dir():
            continue
        di, dl = DS / tag / split / "images", DS / tag / split / "labels"
        di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
        for img in src.iterdir():
            lab = root / split / "labels" / f"{img.stem}.txt"
            keep = [" ".join(["0", *l.split()[1:5]]) for l in (lab.read_text().splitlines() if lab.exists() else [])
                    if l.split() and int(l.split()[0]) in ids]
            (di / img.name).symlink_to(img)
            (dl / f"{img.stem}.txt").write_text("\n".join(keep))
        extra.setdefault(tag, {})[split] = len(list(di.iterdir()))
print("old sets", extra, flush=True)
dirs = lambda split: [str(p) for p in DS.glob(f"*/{split}/images")]
(DS / "ball.yaml").write_text(yaml.safe_dump({"train": dirs("train"), "val": [str(DS / "st2/valid/images")], "names": {0: "ball"}}))

# 2. fine-tune from the current ball model at the crop's native size
EPOCHS = int(os.environ.get("BALL_EPOCHS", "30"))
YOLO(OLD).train(data=str(DS / "ball.yaml"), epochs=EPOCHS, imgsz=960, batch=8, patience=10, device=0,
                project="/kaggle/working/runs", name="st2", exist_ok=True, mosaic=1.0, scale=0.3,
                close_mosaic=5, lr0=0.002, verbose=False, plots=False)
new_w = OUT / "ball_st2.pt"
new_w.write_bytes(Path("/kaggle/working/runs/st2/weights/best.pt").read_bytes())
new = YOLO(str(new_w))

# 3. held-out M7: found near the projected ball / far balls, old vs new, native and 2x
def st2_eval(model, cls, imgsz, conf=0.12):
    n = found = far = 0
    for p, pu, pv, *_ in rows("tasks_m7.tsv"):
        bs = balls(model, cv2.imread(str(p)), imgsz, conf, cls)
        d = [np.hypot(b[0] - pu, b[1] - pv) for b in bs]
        n += 1
        found += any(x <= MAX_DIST for x in d)
        far += sum(x > 2 * MAX_DIST for x in d)
    return {"crops": n, "found": round(found / max(1, n), 3), "far_per_crop": round(far / max(1, n), 3)}

ev = {"labels": kept, "old_sets": extra, "epochs": EPOCHS, "m7": {}}
for name, model, cls in (("coco_yolov8m", coco, [32]), ("ball_yolov8m", old, [0]), ("ball_st2", new, [0])):
    ev["m7"][name] = {f"imgsz{s}": st2_eval(model, cls, s) for s in (960, 1920)}
    print(name, ev["m7"][name], flush=True)
# old DFL / futsal test splits: no forgetting
for tag in extra:
    t = DS / tag / "test" / "images"
    if t.is_dir():
        one = DS / f"t_{tag}.yaml"
        one.write_text(yaml.safe_dump({"train": str(t), "val": str(t), "names": {0: "ball"}}))
        ev.setdefault("old_test_map50", {})[tag] = {
            k: round(float(YOLO(w).val(data=str(one), imgsz=1280, batch=8, device=0, verbose=False, plots=False).box.map50), 3)
            for k, w in (("ball_yolov8m", OLD), ("ball_st2", str(new_w)))}
(OUT / "st2_eval.json").write_text(json.dumps(ev, indent=2))
print(json.dumps(ev, indent=1))
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.iterdir():
        zf.write(p, p.name)
'''


def build() -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = KERNEL_DIR / "courtlens_ball_st2.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_SLUG, "title": "courtlens-ball-st2", "code_file": path.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            DATA_SLUG,
            f"{OWNER}/courtlens-futsal-models",
            "tiagorodrigues26/ball-annotated-3k-yolov5-pytorch",
            "michaelmortenson/football-soccer-ball-detection-dfl",
            f"{OWNER}/courtlens-roboflow-futsal",
        ],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2))
    return path


def upload_data(crops: Path) -> None:
    """crops/ + task lists → private Kaggle dataset (new version if it exists)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    link = DATA_DIR / "crops"
    if not link.exists():
        link.symlink_to(crops.resolve(), target_is_directory=True)
    (DATA_DIR / "keypoints").mkdir(exist_ok=True)
    for kp in (ROOT / "data/eval/soccertrack2/raw").glob("*/*_keypoints.json"):
        (DATA_DIR / "keypoints" / f"{kp.parent.name}.json").write_bytes(kp.read_bytes())
    for name in ("tasks.tsv", "tasks_m7.tsv"):
        (DATA_DIR / name).write_bytes((ROOT / "data/st2_ball" / name).read_bytes())
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps(
        {"title": "courtlens-st2-ball", "id": DATA_SLUG, "licenses": [{"name": "CC-BY-4.0"}]}))
    exists = subprocess.run(["kaggle", "datasets", "status", DATA_SLUG], capture_output=True, text=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-m", "crops", "-r", "zip"] if exists
           else ["kaggle", "datasets", "create", "-r", "zip"])
    subprocess.run([*cmd, "-p", str(DATA_DIR)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", type=Path, help="crops dir → Kaggle dataset")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    print("kernel →", build())
    if args.upload:
        upload_data(args.upload)
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)


if __name__ == "__main__":
    main()
