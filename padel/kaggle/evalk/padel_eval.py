"""Padel eval kernel (Kaggle CPU): everything downstream of detections, never on the local machine.

Mounts padel-code, padel-pt100-prep (labels), padel-gpu-pose (held-out caches), padel-stroke-rgb
(held-out RGB stroke probabilities). Runs pytest, train_events (heldout, --save), run_eval.
Writes /kaggle/working/out/{eval/*.json, models/*_lgbm.*, logs/*.log}.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def first(p):
    h = sorted(glob.glob(p, recursive=True))
    return Path(h[0]) if h else None


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pytest", "httpx", "ultralytics==8.4.189"],
                   check=False)
    code = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    labels = first("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json").parent
    cache = first("/kaggle/input/**/out/cache/2022_BCN_FinalF_1_heldout.npz").parent
    rgb = first("/kaggle/input/**/stroke_rgb_heldout.npz")
    repo = Path("/tmp/padel")
    shutil.copytree(code, repo)
    (repo / "data" / "pt100").mkdir(parents=True)
    shutil.copytree(labels, repo / "data" / "pt100" / "labels")
    (repo / "data" / "kaggle_out" / "gpu" / "out").mkdir(parents=True)
    os.symlink(cache, repo / "data" / "kaggle_out" / "gpu" / "out" / "cache")
    if rgb:
        (repo / "data" / "kaggle_out" / "stroke" / "out").mkdir(parents=True)
        shutil.copy(rgb, repo / "data" / "kaggle_out" / "stroke" / "out" / rgb.name)
    out = Path("/kaggle/working/out")
    for d in ("eval", "models", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(repo))
    for name, cmd in (("pytest", [sys.executable, "-m", "pytest", "-q", "tests"]),
                      ("events", [sys.executable, "scripts/train_events.py", "--source", "heldout", "--save"]),
                      ("eval", [sys.executable, "-m", "padel_analyzer.eval.run_eval"]),
                      ("events_oracle", [sys.executable, "scripts/train_events.py", "--source", "oracle"])):
        with open(out / "logs" / f"{name}.log", "w") as lf:
            rc = subprocess.run(cmd, cwd=repo, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(name, "rc", rc, flush=True)
        print((out / "logs" / f"{name}.log").read_text()[-3000:], flush=True)
    for f in (repo / "data" / "eval").glob("*.json"):
        shutil.copy(f, out / "eval" / f.name)
    for f in (repo / "models").glob("*_lgbm.*"):
        shutil.copy(f, out / "models" / f.name)


if __name__ == "__main__":
    main()
