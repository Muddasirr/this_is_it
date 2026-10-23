#!/usr/bin/env python3
"""Build + push the Kaggle GPU eval: code dataset (real backend) + kernel notebook.

The kernel runs `python -m app.scripts.run_offline` — the same process_job the
production worker runs — so Kaggle numbers are product numbers. There is no
forked copy of the pipeline any more.

  python scripts/kaggle/build_kaggle.py            # build only
  python scripts/kaggle/build_kaggle.py --push     # build + upload code + run kernel
  kaggle kernels status muddasirr/courtlens-gpu-eval
  kaggle kernels output muddasirr/courtlens-gpu-eval -p data/kaggle_out/gpu_eval --file-pattern 'results.zip'
  unzip -o data/kaggle_out/gpu_eval/results.zip -d data/kaggle_out/gpu_eval   # thumbs.zip likewise

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
import time
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
    "umap-learn>=0.5.6",  # team split (transformers/SigLIP ship with the Kaggle image)
    "motmetrics>=1.4.0",  # MOT benchmark (SoccerTrack v2)
    "remotezip>=0.12",  # read SoccerNet GSR members without downloading the 10 GB zips
]


# Same id in the code bundle and the notebook: the kernel refuses to run on a stale bundle
# (Kaggle may mount the previous dataset version if the new one is still processing).
BUILD_ID = time.strftime("%Y%m%d-%H%M%S")


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
        zf.writestr("BUILD_ID", BUILD_ID)
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
mounted = (BACKEND / "BUILD_ID").read_text().strip() if (BACKEND / "BUILD_ID").exists() else "none"
assert mounted == "__BUILD_ID__", f"stale code bundle {{mounted}} (expected __BUILD_ID__) — re-push after the dataset finishes processing"
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

# Corpus config: one general detector for every video (futsal CCTV, futsal broadcast, DFL)
PRIMARY = os.environ.get("COURTLENS_DETECTOR", "yolov8m.pt")  # best on all 5 futsal videos (v6)
env = dict(os.environ, PYTHONPATH=str(BACKEND), VISION_DEVICE="0", VISION_TARGET_FPS="10",
           VISION_IMGSZ=os.environ.get("COURTLENS_IMGSZ", "0"), VISION_JERSEY_OCR="1",
           VISION_REID="1", VISION_TEAM_EMBED="1", YOLO_MODEL=PRIMARY)

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
# Detector comparison on futsal (the football fine-tune finds ~0.2 players/frame on CCTV)
FUTSAL = ["test_video_2.mp4", "portugal_spain_euro_2026_final.mp4", "bangladesh_pakistan_saff_2026.mp4",
          "spain_brazil_fifa_2012.mp4", "spain_france_u19_2023.mp4"]
DETECTORS = [d for d in os.environ.get("COURTLENS_DETECTORS", PRIMARY).split(",") if d]
for name in FUTSAL:
    if name not in mp4:
        continue
    for det in DETECTORS:
        local = next(iter(Path("/kaggle/input").rglob(det)), None)
        index.append(run(f"{name.removesuffix('.mp4')}__{det.removesuffix('.pt')}", {"A": mp4[name]},
                         YOLO_MODEL=str(local or det)))
        (OUT / "eval_index.json").write_text(json.dumps(index, indent=2))

# DFL 30 s clips (tracking / kit-split stress; dataset has no event labels)
dfl = sorted(n for n in mp4 if n[:1] in "ABCDE" and "(" in n)  # e.g. "A1606b0e6_0 (11).mp4"
step = max(1, len(dfl) // max(1, N_DFL)) if N_DFL else 0
for name in (dfl[::step][:N_DFL] if N_DFL else []):
    slug = "dfl_" + name.replace(" ", "").replace("(", "_").replace(")", "").removesuffix(".mp4")
    index.append(run(slug, {"A": mp4[name]}, MIN_VIDEO_DURATION_S="20"))
(OUT / "eval_index.json").write_text(json.dumps(index, indent=2))
'''

MULTICAM = '''
# 3) Synthetic multi-camera check with known truth:
#    B = left-side crop, brighter, starting 3.0 s later → expected offset_ms ≈ +3000
#    C = 1.25× zoom starting 1.5 s later     → expected offset_ms ≈ +1500
# Source: the DFL clip with the most canonical events (so fuse is actually exercised)
dfl_rows = [r for r in index if r["slug"].startswith("dfl_") and r.get("counts")]
best = max(dfl_rows, key=lambda r: r["counts"]["canonical_events"], default=None)
src_name = next((n for n in dfl if best and best["slug"] == "dfl_" + n.replace(" ", "").replace("(", "_").replace(")", "").removesuffix(".mp4")), None)
src = mp4.get(src_name) or mp4.get("B1606b0e6_1 (37).mp4") or (mp4.get(dfl[0]) if dfl else None)
print("multicam source", src)
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


BENCH = '''
# 4) Event ground truth: alejopaullier/dfl-clips (clips cut around DFL competition labels)
clips_root = next((p.parent for p in Path("/kaggle/input").rglob("play-*.mp4")), None)
if clips_root is not None:
    root = clips_root.parents[1] if clips_root.name == "clips" else clips_root
    r = subprocess.run([sys.executable, "-m", "app.scripts.bench_events", "--clips", "/kaggle/input",
                        "--out", str(OUT / "bench_events.json"), "--per-class", "150",
                        "--save-dets", str(OUT / "bench_dets")],
                       cwd=BACKEND, env=dict(env, VISION_JERSEY_OCR="0", VISION_TEAM_EMBED="0"),
                       capture_output=True, text=True)
    (OUT / "bench_events.log").write_text(r.stdout + "\\n" + r.stderr)
    print(r.stdout[-3000:])
else:
    print("dfl-clips not attached")
'''


MOT = '''
# 6) Detection + tracking vs SoccerTrack v2 MOT ground truth (4096x1080 fixed panorama clips)
mot_runs = []
# Kaggle extracts the upload without the mot/ prefix: <id>.txt beside clips/<id>.mp4
for gt in sorted(Path("/kaggle/input").rglob("*.txt")):
    clip = gt.parent / "clips" / f"{gt.stem}.mp4"
    if not clip.exists():
        continue
    out_json = OUT / f"bench_mot_{gt.stem}.json"
    r = subprocess.run([sys.executable, "-m", "app.scripts.bench_mot", "--clip", str(clip), "--gt", str(gt),
                        "--imgsz", os.environ.get("COURTLENS_MOT_IMGSZ", "640,1280,1920"),
                        "--max-frames", "600", "--out", str(out_json),
                        *(["--teams", str(t)] if (t := BACKEND / "app/scripts/bench_labels" / f"{gt.stem}_teams.json").exists() else [])],
                       cwd=BACKEND, env=env, capture_output=True, text=True)
    (OUT / f"bench_mot_{gt.stem}.log").write_text(r.stdout + "\\n" + r.stderr)
    print(r.stdout[-2000:])
'''


MATCH = '''
# 7) Full-match product run vs SoccerTrack v2 labels (events + team split), M2 first half
half = next(iter(Path("/kaggle/input").rglob("*_panorama_1st_half.mp4")), None)
labels = next(iter(Path("/kaggle/input").rglob("*_12_class_events.json")), None)
if half and labels:
    mid = half.name.split("_")[0]
    frames = os.environ.get("COURTLENS_MATCH_FRAMES", "9000")  # processed frames (~12 min at 10 Hz)
    row = run(f"soccertrack_{mid}_1st", {"A": half}, VISION_MAX_FRAMES=frames)
    index.append(row)
    if row.get("rc") == 0:
        r = subprocess.run([sys.executable, "-m", "app.scripts.bench_soccertrack", "--run", str(OUT / f"soccertrack_{mid}_1st"),
                            "--events", str(labels), "--half", "1", "--out", str(OUT / f"bench_soccertrack_{mid}.json")],
                           cwd=BACKEND, env=env, capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-1000:])
    (OUT / "eval_index.json").write_text(json.dumps(index, indent=2))
'''


JERSEY = '''
# 8) Jersey numbers: train on SoccerNet GSR train crops, score vs GSR valid tracks
weights = OUT / "jersey_resnet18.pt"  # inside OUT so results.zip carries it
r = subprocess.run([sys.executable, "-m", "app.scripts.train_jersey", "--out", str(weights),
                    "--sequences", os.environ.get("COURTLENS_JERSEY_SEQS", "40"), "--epochs", "6"],
                   cwd=BACKEND, env=env, capture_output=True, text=True)
(OUT / "train_jersey.log").write_text(r.stdout + "\\n" + r.stderr)
print(r.stdout[-1500:], r.stderr[-1500:])
if weights.exists():
    r = subprocess.run([sys.executable, "-m", "app.scripts.bench_jersey", "--readers", "classifier,tesseract",
                        "--sequences", "6", "--out", str(OUT / "bench_jersey.json")],
                       cwd=BACKEND, env=dict(env, VISION_JERSEY_MODEL=str(weights)), capture_output=True, text=True)
    (OUT / "bench_jersey.log").write_text(r.stdout + "\\n" + r.stderr)
    print(r.stdout[-2500:])
'''


BUNDLE = '''
# 5) One-file bundles — `kaggle kernels output` fetches files one by one (thousands of thumbs)
import zipfile
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob("*"):
        if p.is_file() and (p.suffix in {".json", ".log"} or p.name == "jersey_resnet18.pt"):
            zf.write(p, p.relative_to(OUT.parent))
with zipfile.ZipFile("/kaggle/working/thumbs.zip", "w", zipfile.ZIP_STORED) as zf:
    for p in OUT.rglob("thumbs/*/*.jpg"):
        zf.write(p, p.relative_to(OUT.parent))
print("bundles", Path("/kaggle/working/results.zip").stat().st_size // 1_000_000, "MB")
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
            _cell(SETUP.replace("__BUILD_ID__", BUILD_ID)),
            _cell(GPU),
            _cell(RUN),
            _cell(MULTICAM),
            _cell(BENCH),
            _cell(MOT),
            _cell(MATCH),
            _cell(JERSEY),
            _cell(BUNDLE),
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
                    "alejopaullier/dfl-clips",
                    f"{OWNER}/courtlens-soccertrack-mot",
                    f"{OWNER}/courtlens-soccertrack-match",
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
    # The new version takes a while to process; pushing the kernel earlier mounts the old one.
    for _ in range(60):
        time.sleep(20)
        st = subprocess.run(["kaggle", "datasets", "status", CODE_SLUG], capture_output=True, text=True).stdout
        if "ready" in st.lower():
            break
    time.sleep(30)
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
