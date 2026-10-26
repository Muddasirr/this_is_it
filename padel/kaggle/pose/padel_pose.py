"""Padel GPU kernel #2 (Kaggle T4 x2): pose detectors + held-out caches + downstream eval, all on Kaggle.

Inputs (mounted):
  padel-pt100-prep   videos, labels, YOLO frame sets
  padel-gpu-train    cross-match ball weights (ball_F / ball_M) + production ball_all
  padel-code         this repo's padel_analyzer/, scripts/, kaggle/gpu/padel_gpu.py

Stages (results land in /kaggle/working as they finish):
  1. pose_F / pose_M cross-match (batch 4: batch 8 @1280 OOMs a T4 and corrupts CUDA on retry)
  2. held-out caches: every frame of each final with the ball + pose models trained on the other final
  3. downstream on the caches: hit + stroke models (cross-match + production), run_eval -> eval/*.json
  4. production pose_all
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        sys.exit(f"not mounted: {pattern} :: {os.listdir('/kaggle/input')}")
    return Path(hits[0])


def find_code() -> Path:
    hits = glob.glob("/kaggle/input/**/padel_analyzer/__init__.py", recursive=True)
    if hits:
        return Path(hits[0]).parents[1]
    # dataset uploaded with zipped folders and not auto-extracted
    import zipfile

    dst = Path("/kaggle/working/code")
    for z in glob.glob("/kaggle/input/**/padel-code/**/*.zip", recursive=True) + \
            glob.glob("/kaggle/input/**/padel_analyzer.zip", recursive=True):
        name = Path(z).stem
        with zipfile.ZipFile(z) as f:
            f.extractall(dst / name if name in ("padel_analyzer", "scripts", "kaggle") else dst)
    return find(str(dst) + "/**/padel_analyzer/__init__.py").parents[1]


CODE = find_code()
sys.path.insert(0, str(CODE / "kaggle" / "gpu"))
import padel_gpu as G  # noqa: E402

TRAIN_OUT = find("/kaggle/input/**/out/models/ball_F/best.pt").parents[2]
OUT = Path("/kaggle/working/out")
G.OUT = OUT
REPO = Path("/kaggle/working/padel")  # code + data laid out like the local repo


def setup_repo(prep: Path) -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(CODE, REPO, ignore=shutil.ignore_patterns("__pycache__"))
    lab = REPO / "data" / "pt100" / "labels"
    lab.mkdir(parents=True)
    for f in (prep / "labels").iterdir():
        shutil.copy(f, lab / f.name)  # writable: the GT loader caches .npz next to the json
    cache = REPO / "data" / "kaggle_out" / "gpu" / "out" / "cache"
    cache.parent.mkdir(parents=True)
    os.symlink(OUT / "cache", cache)
    os.symlink(prep / "videos", REPO / "data" / "videos")


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", G.ULTRA, "lightgbm"], check=False)
    for d in ("logs", "models", "cache", "eval"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    prep = G.find_prep()
    G.log("code", CODE, "train_out", TRAIN_OUT, "prep", prep, "gpus", G.n_gpus())
    status: dict[str, object] = {}

    def stage(name, fn):
        t = time.time()
        try:
            status[name] = fn()
        except Exception:  # noqa: BLE001
            status[name] = "FAILED " + traceback.format_exc()[-1500:]
        G.log("stage", name, f"{(time.time() - t) / 60:.1f} min", str(status[name])[:300])
        (OUT / "status.json").write_text(json.dumps(status, indent=2, default=str))

    def pose_cross():
        jobs = [(f"pose_{a}", G.train_cmd("pose", G.make_set(prep, "pose", [a], b), f"pose_{a}", 1.1))
                for a, b in (("F", "M"), ("M", "F"))]
        rc = G.run_parallel(jobs)
        for name in rc:
            G.keep_weights(name)
        return rc

    stage("pose", pose_cross)

    def cache():
        threads, errors = [], []
        for i, (key, other) in enumerate((("F", "M"), ("M", "F"))):
            bw = str(TRAIN_OUT / "models" / f"ball_{other}" / "best.pt")
            pw = str(OUT / "models" / f"pose_{other}" / "best.pt")
            if not os.path.exists(pw):
                return f"missing {pw}"
            npz = OUT / "cache" / f"{G.MATCHES[key]}_heldout.npz"

            def work(key=key, bw=bw, pw=pw, npz=npz, dev=i % max(1, G.n_gpus())):
                try:
                    G.infer_match(key, bw, pw, npz, device=dev)
                except Exception:  # noqa: BLE001
                    errors.append(traceback.format_exc()[-1500:])

            threads.append(threading.Thread(target=work))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors or "ok"

    stage("cache", cache)

    def downstream():
        setup_repo(prep)
        env = dict(os.environ, PYTHONPATH=str(REPO))
        res = {}
        for name, cmd in (("events", [sys.executable, "scripts/train_events.py", "--source", "heldout", "--save"]),
                          ("eval", [sys.executable, "-m", "padel_analyzer.eval.run_eval"]),
                          ("events_oracle", [sys.executable, "scripts/train_events.py", "--source", "oracle"])):
            with open(OUT / "logs" / f"{name}.log", "w") as lf:
                res[name] = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        for f in (REPO / "data" / "eval").glob("*.json"):
            shutil.copy(f, OUT / "eval" / f.name)
        for f in (REPO / "models").glob("*_lgbm.*"):
            shutil.copy(f, OUT / "models" / f.name)
        return res

    stage("downstream", downstream)

    def production():
        rc = G.run_parallel([("pose_all", G.train_cmd("pose", G.make_set(prep, "pose", ["F", "M"], "M"),
                                                      "pose_all", 1.0))])
        G.keep_weights("pose_all")
        return rc

    stage("production", production)
    G.log("final", json.dumps(status, default=str)[:3000])


if __name__ == "__main__":
    main()
