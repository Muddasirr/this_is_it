#!/usr/bin/env python3
"""Build + push the Kaggle GPU eval: code dataset (real backend) + kernel notebook.

The kernel runs `python -m app.scripts.run_offline` — the same process_job the
production worker runs — so Kaggle numbers are product numbers. There is no
forked copy of the pipeline any more.

  python scripts/kaggle/build_kaggle.py            # build only
  python scripts/kaggle/build_kaggle.py --push     # build + upload code + run kernel
  kaggle kernels status muddasirr/courtlens-gpu-eval
  kaggle kernels output muddasirr/courtlens-gpu-eval -p data/kaggle_out/gpu_eval

Inputs attached to the kernel:
  muddasirr/courtlens-code            this repo's backend (zip, rebuilt on every push)
  muddasirr/courtlens-test-video-2    facility CCTV + highlight corpus
  muddasirr/courtlens-yolov5x-football  detector weights
  saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv  DFL 30 s clips (no labels)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CODE_DIR = HERE / "code_dataset"
KERNEL_DIR = HERE / "kernel"
OWNER = "muddasirr"
CODE_SLUG = f"{OWNER}/courtlens-code"
KERNEL_SLUG = f"{OWNER}/courtlens-gpu-eval"
DFL_SLUG = "saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv"

# Pins match the earlier green Kaggle runs; torch/torchvision come from the image.
PIP = [
    "ultralytics==8.4.183",
    "lap>=0.5.12",
    "pydantic-settings==2.6.1",
    "pytesseract>=0.3.10",
    "torchreid==0.2.5",
]


def build_code_zip() -> Path:
    CODE_DIR.mkdir(exist_ok=True)
    dest = CODE_DIR / "courtlens_backend.zip"
    backend = ROOT / "backend"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("app", "cfg"):
            for p in sorted((backend / sub).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    zf.write(p, p.relative_to(backend))
        zf.write(backend / "requirements.txt", "requirements.txt")
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "backend"], cwd=ROOT, capture_output=True, text=True)
        zf.writestr("GIT_REV", (rev.stdout.strip() or "unknown") + ("-dirty" if dirty.stdout.strip() else ""))
    (CODE_DIR / "dataset-metadata.json").write_text(
        json.dumps({"title": "CourtLens code", "id": CODE_SLUG, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    return dest


def _cell(src: str, kind: str = "code") -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


SETUP = f'''
import glob, os, shutil, subprocess, sys, zipfile
from pathlib import Path

# System deps: ffmpeg (probe/sync), tesseract (jersey OCR — was missing in old runs)
subprocess.call("which ffmpeg ffprobe || (apt-get -qq update && apt-get -qq install -y ffmpeg)", shell=True)
subprocess.call("which tesseract || (apt-get -qq update && apt-get -qq install -y tesseract-ocr)", shell=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *{PIP!r}])

# Kaggle auto-extracts uploaded zips; accept either the zip or the extracted tree
BACKEND = Path("/kaggle/working/backend")
if BACKEND.exists():
    shutil.rmtree(BACKEND)
code_zip = glob.glob("/kaggle/input/**/courtlens_backend.zip", recursive=True)
runner = glob.glob("/kaggle/input/**/app/scripts/run_offline.py", recursive=True)
if code_zip:
    zipfile.ZipFile(code_zip[0]).extractall(BACKEND)
elif runner:
    shutil.copytree(Path(runner[0]).parents[2], BACKEND)  # read-only input → writable copy
else:
    raise SystemExit("attach dataset {CODE_SLUG}")
print("code rev", (BACKEND / "GIT_REV").read_text())
'''

GPU = '''
import torch
assert torch.cuda.is_available(), "Settings → Accelerator → GPU T4, then re-run"
print(torch.__version__, torch.cuda.get_device_name(0))
'''

RUN = '''
import json, os, shutil, subprocess, sys, time
from pathlib import Path

N_DFL = int(os.environ.get("COURTLENS_N_DFL", "8"))
OUT = Path("/kaggle/working/eval"); OUT.mkdir(exist_ok=True)
mp4 = {}
for p in Path("/kaggle/input").rglob("*.mp4"):
    mp4.setdefault(p.name, p)
weights = next(iter(Path("/kaggle/input").rglob("yolov5x_football.pt")), None)

env = dict(os.environ, PYTHONPATH=str(BACKEND), VISION_DEVICE="0", VISION_FRAME_STRIDE="5",
           VISION_JERSEY_OCR="1", VISION_REID="1", YOLO_MODEL=str(weights or "yolov8n.pt"))

def run(slug, cams, **extra):
    out = OUT / slug
    cmd = [sys.executable, "-m", "app.scripts.run_offline", "--out", str(out), "--title", slug]
    for label, path in cams.items():
        cmd += ["--camera", f"{label}={path}"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BACKEND, env=dict(env, **extra), capture_output=True, text=True)
    (OUT / f"{slug}.log").write_text(r.stdout + "\\n" + r.stderr)
    row = {"slug": slug, "cameras": sorted(cams), "rc": r.returncode, "seconds": round(time.time() - t0)}
    if r.returncode == 0:
        s = json.loads((out / "summary.json").read_text())
        row.update(counts=s["counts"], sync=s["sync"], vision=s["stages"].get("detect_track", {}).get("vision_stats"))
        for big in ("courtlens.db", "storage"):  # keep the output small
            p = out / big
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    else:
        row["error"] = r.stderr[-2000:]
    print(json.dumps({k: row.get(k) for k in ("slug", "rc", "seconds", "counts", "sync")}))
    return row

index = []
# 1) Product run: facility CCTV, Camera A only
if "test_video_2.mp4" in mp4:
    index.append(run("test_video_2", {"A": mp4["test_video_2.mp4"]}))

# 2) DFL 30 s clips (tracking / kit-split stress; dataset has no event labels)
dfl = sorted(n for n in mp4 if n[:1] in "ABCDE" and "(" in n)  # e.g. "A1606b0e6_0 (11).mp4"
step = max(1, len(dfl) // max(1, N_DFL))
for name in dfl[::step][:N_DFL]:
    slug = "dfl_" + name.replace(" ", "").replace("(", "_").replace(")", "").removesuffix(".mp4")
    index.append(run(slug, {"A": mp4[name]}, MIN_VIDEO_DURATION_S="20"))
(OUT / "eval_index.json").write_text(json.dumps(index, indent=2))
'''

MULTICAM = '''
# 3) Synthetic multi-camera check with known truth:
#    B = left-side crop, brighter, starting 3.0 s later → expected offset_ms ≈ +3000
#    C = 1.25× zoom starting 1.5 s later     → expected offset_ms ≈ +1500
src = mp4.get(dfl[0]) if dfl else mp4.get("test_video_2.mp4")
mc = OUT / "_multicam_src"; mc.mkdir(exist_ok=True)
a, b, c = mc / "A.mp4", mc / "B.mp4", mc / "C.mp4"
enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an"]
subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", "28", *enc, str(a)])
subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-ss", "3", "-i", str(a),
                       "-vf", "crop=iw*0.8:ih*0.8:0:ih*0.1,scale=iw/0.8:ih/0.8,eq=brightness=0.05", *enc, str(b)])
subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-ss", "1.5", "-i", str(a),
                       "-vf", "crop=iw/1.25:ih/1.25,scale=iw*1.25:ih*1.25", *enc, str(c)])
row = run("multicam_synthetic", {"A": a, "B": b, "C": c}, MIN_VIDEO_DURATION_S="20")
expected = {"B": 3000, "C": 1500}
checks = {}
for s in row.get("sync") or []:
    err = abs(s["offset_ms"] - expected[s["camera"]])
    checks[f"sync_{s['camera']}"] = {"offset_ms": s["offset_ms"], "error_ms": err, "pass": err <= 200,
                                    "method": s["method"], "confidence": s["confidence"]}
if row.get("rc") == 0:
    players = json.loads((OUT / "multicam_synthetic" / "players.json").read_text())
    events = json.loads((OUT / "multicam_synthetic" / "canonical_events.json").read_text())
    multi = [p for p in players if len(p["cameras"]) > 1]
    checks["players_linked_across_cameras"] = {"linked": len(multi), "players": len(players),
                                               "pass": bool(players) and len(multi) / len(players) >= 0.5}
    fused = sum(1 for e in events if len(e["cameras"]) > 1)
    checks["events_fused_across_cameras"] = {"fused": fused, "events": len(events),
                                             "pass": not events or fused / len(events) >= 0.5}
row["checks"] = checks
row["pass"] = bool(checks) and all(v["pass"] for v in checks.values())
index.append(row)
(OUT / "eval_index.json").write_text(json.dumps(index, indent=2))
shutil.rmtree(mc)
print(json.dumps(checks, indent=2))
print("MULTICAM", "PASS" if row["pass"] else "FAIL")
'''


def build_kernel() -> Path:
    KERNEL_DIR.mkdir(exist_ok=True)
    for old in KERNEL_DIR.glob("*.ipynb"):
        old.unlink()
    nb = {
        "cells": [
            _cell(
                "# CourtLens GPU eval (generated by scripts/kaggle/build_kaggle.py — do not edit here)\n"
                "Runs the real backend pipeline (`app.scripts.run_offline`) on facility CCTV, DFL clips, "
                "and a synthetic 3-camera check. Accelerator: **GPU T4**.",
                "markdown",
            ),
            _cell(SETUP),
            _cell(GPU),
            _cell(RUN),
            _cell(MULTICAM),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = KERNEL_DIR / "courtlens_gpu_eval.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL_SLUG,
                "title": "courtlens-gpu-eval",
                "code_file": path.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [
                    CODE_SLUG,
                    f"{OWNER}/courtlens-test-video-2",
                    f"{OWNER}/courtlens-yolov5x-football",
                    DFL_SLUG,
                ],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
    )
    return path


def push() -> None:
    exists = subprocess.run(["kaggle", "datasets", "status", CODE_SLUG], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        cmd = ["kaggle", "datasets", "version", "-p", str(CODE_DIR), "-m", "backend update", "-r", "skip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(CODE_DIR), "-r", "skip"]
    subprocess.run(cmd, check=True)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    print("code  →", build_code_zip())
    print("kernel→", build_kernel())
    if args.push:
        push()


if __name__ == "__main__":
    main()
