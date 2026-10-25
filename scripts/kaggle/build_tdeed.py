"""Kaggle kernel: pretrained T-DEED ball-action spotting vs our labelled sets.

T-DEED (Xarles et al., SoccerNet Ball Action Spotting) predicts when PASS / DRIVE /
SHOT / GOAL / ... happen. This kernel runs the published checkpoints on
  - SoccerTrack v2 M2 first half (first 12 min, 167 labelled passes) → bench_soccertrack
  - the futsal highlights with scoreboard goal GT → bench_goals
  - test_video_2 + Bangladesh (no GT; outputs kept for the UI)
and bundles everything into results.zip. Uses the same courtlens-code dataset as
the main kernel (build_kaggle.py must have pushed the current backend).

  python scripts/kaggle/build_tdeed.py --push
  python scripts/import_kaggle_run.py ...   # not needed: results are JSON only
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "data" / "kaggle_tdeed_kernel"
OWNER = "muddasirr"
KERNEL_SLUG = f"{OWNER}/courtlens-tdeed-eval"

GT_FILES = {
    "portugal_spain": "portugal_spain_goals.json",
    "spain_brazil": "spain_brazil_goals.json",
    "spain_france": "spain_france_goals.json",
}

SCRIPT = r'''
import glob, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

OUT = Path("/kaggle/working/tdeed"); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/working/T-DEED")
if not REPO.exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/arturxe2/T-DEED.git", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "wandb", "tabulate", "SoccerNet"], check=True)
for sub in ["SoccerNetBall", "SoccerNet"]:
    src = next((p for p in Path("/kaggle/input").rglob(sub) if p.is_dir() and p.name == sub), None)
    if src:
        shutil.copytree(src, REPO / "checkpoints" / sub, dirs_exist_ok=True)

BACKEND = Path("/kaggle/working/backend")
if not BACKEND.exists():
    code_zip = next(Path("/kaggle/input").rglob("courtlens_backend.zip"), None)
    if code_zip:
        zipfile.ZipFile(code_zip).extractall(BACKEND)
    else:  # Kaggle unpacks uploaded zips
        runner = next(Path("/kaggle/input").rglob("app/scripts/run_offline.py"))
        shutil.copytree(runner.parents[2], BACKEND)
sys.path.insert(0, str(BACKEND))
import app.scripts.bench_goals  # fail fast if the code bundle is stale
GT = __GT__

LABEL_MAP = {"PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "DRIVE": "dribble", "SHOT": "shot", "GOAL": "goal"}

def to25(src, dst, seconds=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-vf", "fps=25,scale=796:448", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(dst)]
    subprocess.run(cmd, check=True)

def spot(name, video, threshold):
    v25 = OUT / f"{name}_25.mp4"
    to25(video, v25, seconds=720 if "soccertrack" in name else None)
    res = REPO / "inference_output" / "results_inference.json"
    if res.exists():
        res.unlink()
    r = subprocess.run([sys.executable, "inference.py", "--model", "SoccerNetBall_challenge2", "--video_path", str(v25),
                        "--inference_threshold", str(threshold)], cwd=REPO, capture_output=True, text=True)
    (OUT / f"{name}.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-6000:])
    if not res.exists():
        print(name, "FAILED", r.stderr[-1500:]); return None
    raw = json.loads(res.read_text())["predictions"]
    events = [{"event_type": LABEL_MAP[p["label"]], "label": p["label"], "ts_ms": int(p["frame"] * 40), "confidence": p["confidence"]}
              for p in raw if p["label"] in LABEL_MAP]
    (OUT / f"{name}_raw.json").write_text(json.dumps(raw))
    (OUT / f"{name}_events.json").write_text(json.dumps(events))
    v25.unlink()
    return raw, events

summary = {}
from app.scripts import bench_soccertrack as BS, bench_goals as BG

# SoccerTrack: sweep confidence thresholds on one inference (raw keeps scores ≥ 0.05)
half = next(iter(Path("/kaggle/input").rglob("*_panorama_1st_half.mp4")), None)
labels = next(iter(Path("/kaggle/input").rglob("*_12_class_events.json")), None)
if half and labels:
    out = spot("soccertrack_117093", half, 0.05)
    if out:
        raw, events = out
        truth = BS.load_truth(labels, 1)
        sweep = {}
        for th in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            pred = [{"type": e["event_type"], "ts_ms": e["ts_ms"], "kit": None} for e in events if e["confidence"] >= th]
            sweep[str(th)] = BS.score(truth, pred, 1000, 720000)["by_class"]
        summary["soccertrack_117093"] = sweep

# Futsal broadcast goals (scoreboard GT)
for v in Path("/kaggle/input").rglob("*.mp4"):
    key = next((k for k in GT if v.name.startswith(k)), None)
    if key is None and not (v.name.startswith("test_video_2") or v.name.startswith("bangladesh")):
        continue
    name = v.stem
    out = spot(name, v, 0.05)
    if not out:
        continue
    raw, events = out
    row = {"counts": {}}
    for th in [0.05, 0.1, 0.15, 0.2]:
        ev = [e for e in events if e["confidence"] >= th]
        row["counts"][str(th)] = {t: sum(1 for e in ev if e["event_type"] == t) for t in ["pass", "dribble", "shot", "goal"]}
        if key:
            row[f"goals@{th}"] = BG.score(GT[key]["events"], ev)
    summary[name] = row
    print(name, json.dumps(row)[:600], flush=True)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1)[:6000])
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob("*"):
        if p.suffix in {".json", ".log"}:
            zf.write(p, p.relative_to(OUT.parent))
'''


def build() -> Path:
    gt_dir = ROOT / "backend" / "data" / "futsal_gt"
    gt = {k: json.loads((gt_dir / f).read_text()) for k, f in GT_FILES.items()}
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    path = KERNEL_DIR / "courtlens_tdeed_eval.py"
    path.write_text(SCRIPT.replace("__GT__", repr(gt)))
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_SLUG,
        "title": "courtlens-tdeed-eval",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            f"{OWNER}/courtlens-code",
            f"{OWNER}/courtlens-tdeed-ckpt",
            f"{OWNER}/courtlens-test-video-2",
            f"{OWNER}/courtlens-soccertrack-match",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    print("kernel →", build())
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)


if __name__ == "__main__":
    main()
