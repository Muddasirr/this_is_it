"""Kaggle kernel: fine-tune the SoccerTrack T-DEED spotter on hand-labelled futsal (roadmap W7.2).

Why: on the labelled futsal phone clip (test_video_2, 0–7:10, 72 passes / 12 shots) the
SoccerTrack model reached pass F1 0.44 — no better than the heuristics (0.40) — while it
scores 0.80 on held-out SoccerTrack: a domain gap, not a bug.

Data (muddasirr/courtlens-futsal-tv2): the clip at the model's input (25 fps, 796x448) +
labels exported from the app's Label page. Time splits: train 0–240 s, val 240–300 s
(epoch selection), test 300–431 s (never trained on). Backbone frozen (T4 16 GB; 72 passes
is too little to retrain it), temporal layers + head fine-tuned from tdeed_best.pt.
Reports pass/shot/dribble P/R/F1 on the test split for the SoccerTrack model and the futsal
fine-tune at several thresholds.

  python scripts/kaggle/build_tdeed_futsal_ft.py --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = ROOT / "data" / "kaggle_tdeed_futsal_ft"
SLUG = "muddasirr/courtlens-tdeed-futsal-ft"

SCRIPT = r'''
import json, os, subprocess, sys, types
from pathlib import Path

OUT = Path("/kaggle/working/out"); OUT.mkdir(exist_ok=True)
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "SoccerNet==0.1.62", "tabulate"])
REPO = "/kaggle/working/tdeed"
if not Path(REPO).exists():
    subprocess.check_call(f"git clone -q https://github.com/arturxe2/T-DEED.git {REPO} && "
                          f"git -C {REPO} checkout -q d953b4de8a2ad941c565e1683c5b07255cbe644e", shell=True)
video = str(next(Path("/kaggle/input").rglob("tv2_796x448_25fps.mp4")))
labels = json.loads(next(Path("/kaggle/input").rglob("labels.json")).read_text())["events"]
weights = str(next(Path("/kaggle/input").rglob("tdeed_best.pt")))
TYPE = {"pass": "PASS", "dribble": "DRIVE", "shot": "SHOT", "goal": "GOAL"}
SPLITS = {"train": (0, 240), "val": (240, 300), "test": (300, 431)}
FPS = 25

frames = Path("/kaggle/working/frames"); lab_root = Path("/kaggle/working/labels")
counts = {}
for name, (a, b) in SPLITS.items():
    d = frames / name; d.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", video,
                           "-q:v", "3", "-start_number", "0", str(d / "frame%d.jpg")])
    counts[name] = len(list(d.glob("*.jpg")))
    ann = [{"gameTime": f"1 - {int((l['ts_ms']/1000 - a)//60)}:{int((l['ts_ms']/1000 - a)%60):02d}",
            "label": TYPE[l["type"]], "position": str(int(l["ts_ms"] - a * 1000)), "team": l.get("team"), "visibility": "visible"}
           for l in labels if a * 1000 <= l["ts_ms"] < b * 1000 and l["type"] in TYPE]
    (lab_root / name).mkdir(parents=True, exist_ok=True)
    (lab_root / name / "Labels-ball.json").write_text(json.dumps({"annotations": ann}))
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", video,
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(Path("/kaggle/working") / f"{name}.mp4")])
print("frames", counts, flush=True)

os.chdir(REPO); sys.path.insert(0, REPO)
base = Path(REPO) / "data" / "soccernetball"
(base / "labels_path.txt").write_text(str(lab_root))
for split, names in (("train", ["train"]), ("val", ["val"]), ("test", ["test"])):
    (base / f"{split}.json").write_text(json.dumps([{"video": n, "num_frames": counts[n]} for n in names]))
cfg = json.loads((Path(REPO) / "config/SoccerNetBall/SoccerNetBall_challenge2.json").read_text())
cfg.update(frame_dir=str(frames), save_dir="/kaggle/working/runs", store_dir="/kaggle/working/store", store_mode="store",
           epoch_num_frames=6000, num_epochs=int(os.environ.get("FT_EPOCHS", "15")), warm_up_epochs=1,
           learning_rate=0.0002, criterion="loss", only_test=False, start_val_epoch=999, num_workers=2, batch_size=1)
cfg.pop("pretrain", None)
(Path(REPO) / "config" / "Futsal").mkdir(exist_ok=True)
(Path(REPO) / "config/Futsal/Futsal_ft.json").write_text(json.dumps(cfg))

import torch
from torch.utils.data import DataLoader
from dataset.datasets import get_datasets
from dataset.frame import ActionSpotInferenceDataset
from model.model import TDEEDModel
from util.dataset import load_classes
from util.eval import inference
from util.io import load_json
import train_tdeed as T

os.environ["WANDB_MODE"] = "disabled"
args = T.update_args(types.SimpleNamespace(model="Futsal_ft", seed=1, acc_grad_iter=1), load_json("config/Futsal/Futsal_ft.json"))
args.crop_dim = None
get_datasets(args); args.store_mode = "load"
classes, _, train_data, val_data, _ = get_datasets(args)

def load_model(path):
    m = TDEEDModel(args=args)
    st = torch.load(path, map_location="cpu", weights_only=False)
    st = st["model"] if isinstance(st, dict) and "model" in st else st
    own = m._model.state_dict()
    m._model.load_state_dict({k: v for k, v in st.items() if k in own and own[k].shape == v.shape}, strict=False)
    return m

model = load_model(weights)
for p in model._model._features.parameters():  # frozen backbone (see module doc)
    p.requires_grad = False
optimizer, scaler = model.get_optimizer({"lr": args.learning_rate})
loader = lambda d: DataLoader(d, shuffle=False, batch_size=1, pin_memory=True, num_workers=2, prefetch_factor=2)
tl, vl = loader(train_data), loader(val_data)
_, sched = T.get_lr_scheduler(args, optimizer, len(tl))
best, history = float("inf"), []
for epoch in range(args.num_epochs):
    t = model.epoch(tl, optimizer, scaler, lr_scheduler=sched)
    v = model.epoch(vl)
    history.append({"epoch": epoch, "train": float(t), "val": float(v)}); print(history[-1], flush=True)
    if v < best:
        best = v; torch.save(model._model.state_dict(), OUT / "tdeed_futsal.pt")
(OUT / "history.json").write_text(json.dumps(history, indent=1))

def predictions(path):
    m = load_model(path)
    torch.cuda.empty_cache()
    ds = ActionSpotInferenceDataset("/kaggle/working/test.mp4", clip_len=args.clip_len, overlap_len=args.clip_len // 4 * 3,
                                    stride=2, dataset="soccernetball", size=(796, 448))
    o = Path("inference_output/results_inference.json"); o.unlink(missing_ok=True)
    inference(m, DataLoader(ds, batch_size=1, num_workers=0), load_classes("data/soccernetball/class.txt"), threshold=0.03)
    return json.loads(o.read_text())["predictions"]

MAP = {"PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "DRIVE": "dribble", "SHOT": "shot", "GOAL": "goal"}
a, b = SPLITS["test"]
truth = [l for l in labels if a * 1000 <= l["ts_ms"] < b * 1000]

def match(t, p, tol=1000):
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(t) for j, y in enumerate(p) if abs(x - y) <= tol)
    ut, up, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in ut and j not in up:
            ut.add(i); up.add(j); n += 1
    return n

def score(raw):
    out = {}
    for th in (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4):
        row = {}
        for cls in ("pass", "dribble", "shot"):
            tt = [l["ts_ms"] - a * 1000 for l in truth if l["type"] == cls]
            pp = [int(r["frame"] * 40) for r in raw if MAP.get(r["label"]) == cls and r["confidence"] >= th]
            tp = match(tt, pp); P = tp / len(pp) if pp else None; R = tp / len(tt) if tt else None
            row[cls] = {"truth": len(tt), "pred": len(pp), "tp": tp, "P": P and round(P, 3), "R": R and round(R, 3),
                        "F1": round(2 * P * R / (P + R), 3) if P and R else None}
        out[str(th)] = row
    return out

res = {"splits": SPLITS, "frames": counts, "history": history,
       "soccertrack_model": score(predictions(weights)), "futsal_finetune": score(predictions(str(OUT / "tdeed_futsal.pt")))}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
for k in ("soccertrack_model", "futsal_finetune"):
    for th, r in res[k].items():
        print(k, th, "pass", r["pass"], "shot", r["shot"])
'''


def build() -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    path = KERNEL_DIR / "tdeed_futsal_ft.py"
    path.write_text(SCRIPT)
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": SLUG, "title": SLUG.split("/")[1], "code_file": path.name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["muddasirr/courtlens-futsal-tv2", "muddasirr/courtlens-tdeed-weights"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
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
