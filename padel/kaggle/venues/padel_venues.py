"""Padel GPU kernel: venue diversity (Roboflow Universe padel sets, CC BY 4.0) + padel shot-type detector.

Mounted: padel-roboflow (YOLO exports), padel-pt100-prep (PT100 YOLO frames), padel-gpu-train (ball_all),
padel-gpu-pose (pose_all), padel-code (kaggle/gpu/padel_gpu.py helpers).

  1. baseline   production ball / player models on every Roboflow test split (unseen courts + cameras)
  2. ball_v2    fine-tuned from ball_all on PT100 + all Roboflow ball train splits          (GPU 0)
     ball_lovo  same, *without* padel-dataset (ll7pp) -> scored on ll7pp test = unseen venue  (GPU 1)
  3. shots      YOLO11s on padelcnn/padel-quret: per-frame padel shot types (bandeja, volleys, lob, ...)
  4. after      ball_v2 / ball_lovo on every test split
Writes /kaggle/working/out/{venues.json, models/*, logs/*}.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def first(p: str) -> Path | None:
    h = sorted(glob.glob(p, recursive=True))
    return Path(h[0]) if h else None


CODE = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
sys.path.insert(0, str(CODE / "kaggle" / "gpu"))
import padel_gpu as G  # noqa: E402

OUT = Path("/kaggle/working/out")
G.OUT = OUT
WORK = Path("/tmp/venues")
RF = Path("/tmp/roboflow")


def fetch_roboflow() -> None:
    """Download the Roboflow YOLO exports on Kaggle (signed per-export links; no API key leaves the laptop)."""
    import io
    import urllib.request
    import zipfile

    links = json.loads((CODE / "kaggle" / "venues" / "roboflow_links.json").read_text())
    for name, d in links.items():
        out = RF / name
        if (out / "data.yaml").exists():
            continue
        t = time.time()
        data = urllib.request.urlopen(d["url"], timeout=3600).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(out)
        G.log("roboflow", name, f"{len(data) / 1e6:.0f} MB", f"{time.time() - t:.0f}s")
BALL_SETS = ["ll7pp", "home", "rbqpp", "rbada", "ptrack"]
PLAYER_SETS = ["ll7pp", "home"]
LOVO = "ll7pp"


def names_of(ds: str) -> list[str]:
    import yaml

    y = yaml.safe_load((RF / ds / "data.yaml").read_text())
    n = y["names"]
    return list(n.values()) if isinstance(n, dict) else list(n)


def split_dir(ds: str, split: str) -> Path | None:
    for cand in (split, "valid" if split == "val" else split):
        d = RF / ds / cand / "images"
        if d.exists():
            return d
    return None


def single_class(ds: str, split: str, keep: str, dst: Path, limit: int | None = None) -> int:
    """Copy one split keeping only class `keep` (renumbered to 0). Returns image count."""
    src = split_dir(ds, split)
    if src is None:
        return 0
    cls = [i for i, n in enumerate(names_of(ds)) if n.lower() == keep]
    if not cls:
        return 0
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    imgs = sorted(src.iterdir())
    if limit:
        random.Random(0).shuffle(imgs)
        imgs = imgs[:limit]
    for im in imgs:
        lab = src.parent / "labels" / (im.stem + ".txt")
        lines = []
        if lab.exists():
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if p and int(p[0]) in cls:
                    lines.append(" ".join(["0"] + p[1:5]))
        os.symlink(im, dst / "images" / f"{ds}_{im.name}")
        (dst / "labels" / f"{ds}_{im.stem}.txt").write_text("\n".join(lines))
    return len(imgs)


def yaml_for(root: Path, train: str, val: str, name: str) -> Path:
    y = root / "data.yaml"
    y.write_text(f"path: {root}\ntrain: {train}\nval: {val}\nnames:\n  0: {name}\n")
    return y


def val_model(weights: str, data: Path, imgsz: int, device: int) -> dict:
    from ultralytics import YOLO

    r = YOLO(weights).val(data=str(data), imgsz=imgsz, batch=8, device=device, verbose=False, plots=False,
                          conf=0.001, iou=0.6, split="val")
    return {"P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4), "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4)}


def test_sets(kind: str, sets: list[str]) -> dict[str, Path]:
    out = {}
    for ds in sets:
        root = WORK / f"test_{kind}_{ds}"
        if not root.exists():
            n = single_class(ds, "test", kind, root / "val")
            if n == 0:
                continue
        out[ds] = yaml_for(root, "val/images", "val/images", kind)
    return out


def build_ball_train(exclude: str | None) -> Path:
    root = WORK / f"ball_train_{exclude or 'all'}"
    if root.exists():
        shutil.rmtree(root)
    n = 0
    for ds in BALL_SETS:
        if ds == exclude:
            continue
        n += single_class(ds, "train", "ball", root / "train")
        single_class(ds, "valid", "ball", root / "val", limit=150)
    # PT100 broadcast frames keep the original domain in the mix (every 2nd sampled frame)
    prep = G.find_prep()
    for m in G.MATCHES.values():
        labs = sorted((prep / "yolo" / "ball_labels" / m).glob("*.txt"))[::2]
        for t in labs:
            os.symlink(prep / "yolo" / "images" / m / (t.stem + ".jpg"), root / "train" / "images" / (t.stem + ".jpg"))
            shutil.copy(t, root / "train" / "labels" / t.name)
        n += len(labs)
    G.log("ball train", exclude, n, "images")
    return yaml_for(root, "train/images", "val/images", "ball")


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", G.ULTRA], check=False)
    for d in ("logs", "models"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    fetch_roboflow()
    ball_all = str(first("/kaggle/input/**/models/ball_all/best.pt"))
    pose_all = str(first("/kaggle/input/**/models/pose_all/best.pt"))
    res: dict = {"sets": {ds: names_of(ds) for ds in BALL_SETS + ["quret"]}}

    def save():
        (OUT / "venues.json").write_text(json.dumps(res, indent=1))

    def stage(name, fn):
        t = time.time()
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001
            res[name] = "FAILED " + traceback.format_exc()[-1500:]
        G.log("stage", name, f"{(time.time() - t) / 60:.1f} min", str(res[name])[:400])
        save()

    tb, tp = test_sets("ball", BALL_SETS), test_sets("player", PLAYER_SETS)

    def baseline():
        out = {}
        for ds, y in tb.items():
            out[f"ball_{ds}"] = {f"imgsz{s}": val_model(ball_all, y, s, 0) for s in (640, 1280)}
        for ds, y in tp.items():
            out[f"player_{ds}"] = {f"imgsz{s}": val_model(pose_all, y, s, 0) for s in (640, 1280)}
        return out

    stage("baseline", baseline)

    def train_ball():
        jobs = []
        for name, excl in (("ball_v2", None), ("ball_lovo", LOVO)):
            y = build_ball_train(excl)
            cmd = [*G.YOLO_CLI, "detect", "train", f"model={ball_all}", f"data={y}", "imgsz=1280", "batch=8",
                   "epochs=60", "time=2.5", "patience=15", "mosaic=0.5", "close_mosaic=5", "scale=0.4",
                   "fliplr=0.5", "workers=4", "plots=False", f"project={G.WORK / 'runs'}", f"name={name}",
                   "exist_ok=True"]
            jobs.append((name, cmd))
        rc = G.run_parallel(jobs)
        for name in rc:
            G.keep_weights(name)
        return rc

    stage("train_ball", train_ball)

    def shots():
        # padelcnn/padel-quret: boxes labelled with the padel shot being played (Bandeja, Remate, Globo, ...)
        y = RF / "quret" / "data.yaml"
        root = WORK / "quret"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(RF / "quret", root, symlinks=True)
        txt = (root / "data.yaml").read_text()
        import re

        txt = re.sub(r"^(train|val|test):.*$", "", txt, flags=re.M)
        txt = f"path: {root}\ntrain: train/images\nval: valid/images\ntest: test/images\n" + txt
        (root / "data.yaml").write_text(txt)
        cmd = [*G.YOLO_CLI, "detect", "train", "model=yolo11s.pt", f"data={root / 'data.yaml'}", "imgsz=640",
               "batch=32", "epochs=80", "time=1.5", "patience=20", "fliplr=0.5", "workers=4", "plots=False",
               f"project={G.WORK / 'runs'}", "name=shot_types", "exist_ok=True"]
        rc = G.run_parallel([("shot_types", cmd)])
        G.keep_weights("shot_types")
        from ultralytics import YOLO

        w = G.WORK / "runs" / "shot_types" / "weights" / "best.pt"
        r = YOLO(str(w)).val(data=str(root / "data.yaml"), split="test", imgsz=640, verbose=False, plots=False)
        names = r.names
        per = {names[int(c)]: round(float(m), 4) for c, m in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}
        return {"rc": rc, "test_mAP50": round(float(r.box.map50), 4), "test_mAP50_95": round(float(r.box.map), 4),
                "per_class_mAP50_95": per}

    stage("shot_types", shots)

    def after():
        out = {}
        for name in ("ball_v2", "ball_lovo"):
            w = G.WORK / "runs" / name / "weights" / "best.pt"
            if w.exists():
                out[name] = {ds: val_model(str(w), y, 1280, 0) for ds, y in tb.items()}
        return out

    stage("after", after)
    G.log("final", json.dumps(res)[:3000])


if __name__ == "__main__":
    main()
