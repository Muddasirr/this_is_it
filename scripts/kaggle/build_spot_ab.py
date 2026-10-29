"""Kaggle kernel: run the SoccerTrack T-DEED spotter on pre-built model-input videos (geometry A/B).

Input dataset muddasirr/courtlens-spot-ab: <variant>.mp4 (796x448, 25 fps) from
scripts/eval/spot_geometry_ab.py. Same inference call as app.services.action_spotting.spot
(clip 100, overlap 75, stride 2, threshold 0.05). Output: results.zip with spotted_<variant>.json
([{ts_ms, label, confidence}]) — score with scripts/eval/score_soccertrack.py.

  python scripts/kaggle/build_spot_ab.py --upload data/st2_ball/spot_ab --push
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER = "muddasirr"
KERNEL_DIR = ROOT / "data" / "kaggle_spot_ab_kernel"
KERNEL_SLUG = f"{OWNER}/courtlens-spot-geometry"
DATA_SLUG = f"{OWNER}/courtlens-spot-ab"
TDEED_COMMIT = "d953b4de8a2ad941c565e1683c5b07255cbe644e"  # = backend kaggle_backend.TDEED_COMMIT

CODE = r'''
import json, os, subprocess, sys, types, zipfile
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "SoccerNet==0.1.62", "tabulate", "timm"], check=True)
work = Path("/kaggle/working")
subprocess.run(f"git clone -q https://github.com/arturxe2/T-DEED.git {work}/tdeed && git -C {work}/tdeed checkout -q __COMMIT__",
               shell=True, check=True)
weights = str(next(Path("/kaggle/input").rglob("tdeed_soccertrack.pt")))
videos = sorted(Path("/kaggle/input").rglob("*.mp4"))
os.chdir(work / "tdeed"); sys.path.insert(0, str(work / "tdeed"))
import torch
from torch.utils.data import DataLoader
from model.model import TDEEDModel
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes
from util.eval import inference

CONFIG = {"clip_len": 100, "feature_arch": "rny008_gsf", "temporal_arch": "ed_sgp_mixer", "n_layers": 2,
          "sgp_ks": 9, "sgp_r": 4, "num_classes": 12, "radi_displacement": 4, "modality": "rgb", "crop_dim": None,
          "dataset": "soccernetball", "batch_size": 4, "num_workers": 0, "pretrain": None}
model = TDEEDModel(device="cuda", args=types.SimpleNamespace(**CONFIG))
ck = torch.load(weights, map_location="cpu", weights_only=False)
model._model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
classes = load_classes(os.path.join("data", "soccernetball", "class.txt"))
out_dir = work / "out"; out_dir.mkdir(exist_ok=True)
for v in videos:
    ds = ActionSpotInferenceDataset(str(v), clip_len=100, overlap_len=75, stride=2, dataset="soccernetball", size=(796, 448))
    res = Path("inference_output/results_inference.json"); res.unlink(missing_ok=True)
    inference(model, DataLoader(ds, batch_size=1, num_workers=0), classes, threshold=0.05)
    raw = json.loads(res.read_text())["predictions"]
    spotted = [{"ts_ms": int(round(p["frame"] * 1000 / 25)), "label": p["label"], "confidence": round(float(p["confidence"]), 3)}
               for p in raw]
    (out_dir / f"spotted_{v.stem}.json").write_text(json.dumps(spotted))
    print(v.stem, len(spotted), flush=True)
with zipfile.ZipFile(work / "results.zip", "w") as zf:
    for p in out_dir.iterdir():
        zf.write(p, p.name)
'''.replace("__COMMIT__", TDEED_COMMIT)


def build() -> Path:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = KERNEL_DIR / "courtlens_spot_ab.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_SLUG, "title": "courtlens-spot-geometry", "code_file": path.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATA_SLUG, f"{OWNER}/courtlens-futsal-models"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2))
    return path


def upload(src: Path) -> None:
    d = ROOT / "data" / "kaggle_spot_ab_dataset"
    d.mkdir(parents=True, exist_ok=True)
    for mp4 in src.glob("*.mp4"):
        link = d / mp4.name
        if not link.exists():
            link.symlink_to(mp4.resolve())
    (d / "dataset-metadata.json").write_text(json.dumps({"title": "courtlens-spot-ab", "id": DATA_SLUG,
                                                          "licenses": [{"name": "CC-BY-4.0"}]}))
    exists = subprocess.run(["kaggle", "datasets", "status", DATA_SLUG], capture_output=True, text=True).returncode == 0
    cmd = ["kaggle", "datasets", "version", "-m", "videos"] if exists else ["kaggle", "datasets", "create"]
    subprocess.run([*cmd, "-p", str(d)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", type=Path)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    print("kernel →", build())
    if args.upload:
        upload(args.upload)
    if args.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)


if __name__ == "__main__":
    main()
