"""Padel GPU kernel (Kaggle T4 x2): train ball + player-pose detectors, cache held-out detections.

Input: output of the CPU kernel `padel-pt100-prep` (videos, labels, sampled YOLO frames).

Stages (each writes into /kaggle/working/out as it finishes, so a late failure keeps earlier work):
  1. cross-match ball detectors   ball_F (train women's final) / ball_M (train men's final)
  2. cross-match pose detectors   pose_F / pose_M
  3. held-out inference cache     every frame of match X with the models trained on the other match
                                  + off-the-shelf yolo11m-pose baseline on every 10th frame
  4. production models            ball_all / pose_all trained on both finals
Downstream (hits, shot types, rallies, stats) trains on CPU from the caches.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ULTRA = "ultralytics==8.4.189"
MATCHES = {"F": "2022_BCN_FinalF_1", "M": "2022_BCN_FinalM_1"}
OUT = Path("/kaggle/working/out")
WORK = Path("/tmp/padel")
BALL_IMGSZ = 1536
POSE_IMGSZ = 1280
YOLO_CLI = [sys.executable, "-c", "from ultralytics.cfg import entrypoint; entrypoint()"]
STAGE_HOURS = {"ball": 2.2, "pose": 1.0, "all_ball": 1.6, "all_pose": 0.8}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_prep() -> Path:
    hits = glob.glob("/kaggle/input/**/pt100/videos", recursive=True)
    if not hits:
        sys.exit("prep output not mounted: " + str(glob.glob("/kaggle/input/*")))
    return Path(hits[0]).parent


def n_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def make_set(prep: Path, kind: str, train: list[str], val: str, n_val: int = 400) -> Path:
    """YOLO dataset of symlinked frames (inputs are read-only) + copied label txts."""
    name = f"{kind}_{''.join(train)}"
    root = WORK / name
    if root.exists():
        shutil.rmtree(root)
    lab_root = prep / "yolo" / f"{kind}_labels"
    for split, keys in (("train", train), ("val", [val])):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for k in keys:
            txts = sorted((lab_root / MATCHES[k]).glob("*.txt"))
            if split == "val":
                txts = txts[:: max(1, len(txts) // n_val)]
            for t in txts:
                img = prep / "yolo" / "images" / MATCHES[k] / (t.stem + ".jpg")
                os.symlink(img, root / "images" / split / img.name)
                shutil.copy(t, root / "labels" / split / t.name)
    yaml = root / "data.yaml"
    lines = [f"path: {root}", "train: images/train", "val: images/val"]
    if kind == "pose":
        lines += ["kpt_shape: [17, 3]", "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]",
                  "names:", "  0: player"]
    else:
        lines += ["names:", "  0: ball"]
    yaml.write_text("\n".join(lines) + "\n")
    log(name, "train", len(list((root / "images/train").iterdir())), "val", len(list((root / "images/val").iterdir())))
    return yaml


def train_cmd(kind: str, yaml: Path, name: str, hours: float) -> list[str]:
    common = [f"data={yaml}", f"project={WORK / 'runs'}", f"name={name}", "exist_ok=True", f"time={hours}",
              "patience=20", "workers=4", "plots=False", "val=True"]
    if kind == "ball":
        return [*YOLO_CLI, "detect", "train", "model=yolo11s.pt", f"imgsz={BALL_IMGSZ}", "batch=8", "epochs=80",
                "mosaic=0.5", "close_mosaic=5", "scale=0.3", "translate=0.05", "fliplr=0.5", "degrees=0",
                "box=7.5", "cls=0.5", *common]
    return [*YOLO_CLI, "pose", "train", "model=yolo11m-pose.pt", f"imgsz={POSE_IMGSZ}", "batch=4", "epochs=60",
            "mosaic=0.5", "close_mosaic=5", "scale=0.3", "fliplr=0.5", *common]


def run_parallel(jobs: list[tuple[str, list[str]]]) -> dict[str, int]:
    """Run jobs on separate GPUs when two are present, otherwise one after another."""
    g = max(1, n_gpus())
    rc: dict[str, int] = {}
    pending = list(jobs)
    while pending:
        batch, pending = pending[:g], pending[g:]
        procs = []
        for i, (name, cmd) in enumerate(batch):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            logf = open(OUT / "logs" / f"{name}.log", "w")
            log("start", name, " ".join(cmd[:4]))
            procs.append((name, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
        for name, p, logf in procs:
            rc[name] = p.wait()
            logf.close()
            log("done", name, "rc", rc[name])
    return rc


def keep_weights(run: str) -> None:
    src = WORK / "runs" / run
    dst = OUT / "models" / run
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("weights/best.pt", "results.csv", "args.yaml"):
        if (src / f).exists():
            shutil.copy(src / f, dst / Path(f).name)


def infer_match(key: str, ball_w: str, pose_w: str, out_npz: Path, device: int = 0, base_every: int = 10) -> None:
    """Every-frame detections for one match (one thread per GPU)."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    prep = find_prep()
    cap = cv2.VideoCapture(str(prep / "videos" / f"{MATCHES[key]}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_m, pose_m, base_m = YOLO(ball_w), YOLO(pose_w), YOLO("yolo11m-pose.pt")
    ball = np.zeros((n, 5, 5), np.float32)
    pbox = np.zeros((n, 10, 5), np.float32)
    pkpt = np.zeros((n, 10, 17, 3), np.float16)
    bbox_ = np.zeros((n, 10, 5), np.float32)
    bkpt = np.zeros((n, 10, 17, 3), np.float16)
    B = 8
    f0 = 0
    t0 = time.time()
    while f0 < n:
        frames = []
        for _ in range(B):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        if not frames:
            break
        for j, r in enumerate(ball_m.predict(frames, imgsz=BALL_IMGSZ, conf=0.03, max_det=5, verbose=False, half=True, device=device)):
            b = r.boxes
            k = len(b)
            if k:
                ball[f0 + j, :k, :4] = b.xyxy.cpu().numpy()
                ball[f0 + j, :k, 4] = b.conf.cpu().numpy()
        for j, r in enumerate(pose_m.predict(frames, imgsz=POSE_IMGSZ, conf=0.2, max_det=10, verbose=False, half=True, device=device)):
            k = len(r.boxes)
            if k:
                pbox[f0 + j, :k, :4] = r.boxes.xyxy.cpu().numpy()
                pbox[f0 + j, :k, 4] = r.boxes.conf.cpu().numpy()
                pkpt[f0 + j, :k] = r.keypoints.data.cpu().numpy()
        sel = [j for j in range(len(frames)) if (f0 + j) % base_every == 0]
        if sel:
            for j, r in zip(sel, base_m.predict([frames[j] for j in sel], imgsz=POSE_IMGSZ, conf=0.2, max_det=10,
                                                verbose=False, half=True, device=device)):
                k = len(r.boxes)
                if k:
                    bbox_[f0 + j, :k, :4] = r.boxes.xyxy.cpu().numpy()
                    bbox_[f0 + j, :k, 4] = r.boxes.conf.cpu().numpy()
                    bkpt[f0 + j, :k] = r.keypoints.data.cpu().numpy()
        f0 += len(frames)
        if f0 % 4000 < B:
            log(key, f0, "/", n, f"{f0 / (time.time() - t0):.1f} fps")
    np.savez_compressed(out_npz, ball=ball, pose_box=pbox, pose_kpt=pkpt, base_box=bbox_, base_kpt=bkpt,
                        n_frames=n, ball_w=ball_w, pose_w=pose_w)
    log("saved", out_npz)


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", ULTRA], check=False)
    for d in ("logs", "models", "cache"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    prep = find_prep()
    log("prep at", prep, "gpus", n_gpus())
    status: dict[str, object] = {}
    W = lambda run: str(WORK / "runs" / run / "weights" / "best.pt")  # noqa: E731

    def stage(name, fn):
        t = time.time()
        try:
            status[name] = fn()
        except Exception:  # noqa: BLE001
            status[name] = "FAILED " + traceback.format_exc()[-800:]
        log("stage", name, f"{(time.time() - t) / 60:.1f} min", str(status[name])[:200])
        (OUT / "status.json").write_text(json.dumps(status, indent=2, default=str))

    def cross(kind, hours):
        jobs = [(f"{kind}_{a}", train_cmd(kind, make_set(prep, kind, [a], b), f"{kind}_{a}", hours))
                for a, b in (("F", "M"), ("M", "F"))]
        rc = run_parallel(jobs)
        for name in rc:
            keep_weights(name)
        return rc

    stage("ball", lambda: cross("ball", STAGE_HOURS["ball"]))
    stage("pose", lambda: cross("pose", STAGE_HOURS["pose"]))

    def cache():
        import threading

        threads, errors = [], []
        for i, (key, other) in enumerate((("F", "M"), ("M", "F"))):
            bw, pw = W(f"ball_{other}"), W(f"pose_{other}")
            if not (os.path.exists(bw) and os.path.exists(pw)):
                return f"missing weights for {key}"
            dev = i % max(1, n_gpus())
            npz = OUT / "cache" / f"{MATCHES[key]}_heldout.npz"

            def work(key=key, bw=bw, pw=pw, npz=npz, dev=dev):
                try:
                    infer_match(key, bw, pw, npz, device=dev)
                except Exception:  # noqa: BLE001
                    errors.append(traceback.format_exc()[-800:])

            threads.append(threading.Thread(target=work))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors or "ok"

    stage("cache", cache)

    def production():
        jobs = [("ball_all", train_cmd("ball", make_set(prep, "ball", ["F", "M"], "M"), "ball_all",
                                       STAGE_HOURS["all_ball"])),
                ("pose_all", train_cmd("pose", make_set(prep, "pose", ["F", "M"], "M"), "pose_all",
                                       STAGE_HOURS["all_pose"]))]
        rc = run_parallel(jobs)
        for name in rc:
            keep_weights(name)
        return rc

    stage("production", production)
    log("final", json.dumps(status, default=str)[:2000])


if __name__ == "__main__":
    main()
