"""Fine-tune T-DEED (ball action spotting) on SoccerTrack v2 — roadmap W7.1.

Why: pretrained T-DEED (SoccerNet broadcast) scored pass F1 0.11 on SoccerTrack and
0/12 futsal goals: its 796x448 input squashes the 4096x1080 panorama until players
are ~5 px. SoccerTrack is the closest labelled data to a facility camera (fixed,
whole pitch, 21k labelled ball actions), so we train on it at a usable scale:
a 1920x1080 window that follows the play (SoccerTrack's ball track through the
match homography, smoothed ±2 s), resized to 796x448 — players then look like
on a broadcast / futsal CCTV frame.

Caveat: the crop follows the *annotated* ball track (where, not when). Scores on
held-out halves are therefore an upper bound until the product feeds its own
ball/player centroid (same crop rule).

Splits (dataset README): train M1 M3 M4 M5 M6 M8; held-out M2 (117093, our
existing benchmark half); test M7 M9 M10.

  modal run scripts/modal/tdeed_soccertrack.py --step prep      # 20 halves in parallel (CPU)
  modal run --detach scripts/modal/tdeed_soccertrack.py --step train
  modal run scripts/modal/tdeed_soccertrack.py --step eval
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

# Inside the Modal container this file sits at /root/<name>.py: the repo root only matters locally (image build)
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")
REPO = "/root/tdeed"
HF_REPO = "atomscott/soccertrack-v2"
MATCHES = {"117092": "train", "117093": "heldout", "118575": "train", "118576": "train", "118577": "train",
           "118578": "train", "128057": "test", "128058": "train", "132831": "test", "132877": "test"}
HALVES = [(m, h) for m in MATCHES for h in ("1st", "2nd")]
W, H = 796, 448

app = modal.App("courtlens-tdeed")
vol = modal.Volume.from_name("courtlens", create_if_missing=True)
hf = modal.Secret.from_name("courtlens-hf")
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("numpy<2", "scipy<1.15", "opencv-python-headless==4.10.0.84", "huggingface_hub==0.26.2")
)
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.3.1", "torchvision==0.18.1", "timm==1.0.3", "numpy<2", "opencv-python-headless==4.10.0.84",
                 "tabulate", "tqdm", "SoccerNet", "wandb", "huggingface_hub==0.26.2")  # timm backbone weights
    .run_commands(f"git clone -q --depth 1 https://github.com/arturxe2/T-DEED.git {REPO}")
    .add_local_dir(str(ROOT / "backend" / "app" / "scripts"), "/root/courtlens_scripts")
)


def _half_name(match: str, half: str) -> str:
    return f"{match}_{half}"


@app.function(image=prep_image, secrets=[hf], volumes={"/mnt": vol}, cpu=8, memory=16384,
              timeout=4 * 3600)
def prep_half(match: str, half: str) -> dict:
    """Panorama half → /mnt/data/st/crops/<id>_<half>.mp4 (796x448, 25 fps) + labels/<id>_<half>/Labels-ball.json."""
    import cv2
    import numpy as np
    from huggingface_hub import hf_hub_download

    name = _half_name(match, half)
    out_mp4 = Path(f"/mnt/data/st/crops/{name}.mp4")
    lab_dir = Path(f"/mnt/data/st/labels/{name}")
    if out_mp4.exists() and (lab_dir / "Labels-ball.json").exists():
        return {"half": name, "skipped": True}
    tok = os.environ["HF_TOKEN"]
    get = lambda p: hf_hub_download(HF_REPO, p, repo_type="dataset", token=tok, local_dir="/tmp/hf")  # noqa: E731
    video = get(f"videos/{match}/{match}_panorama_{half}_half.mp4")
    ball = np.load(get(f"ball/{match}_{half}_ball.npz"))
    keypoints = json.loads(Path(get(f"raw/{match}/{match}_keypoints.json")).read_text())
    events = json.loads(Path(get(f"bas/{match}/{match}_12_class_events.json")).read_text())

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Crop centre per frame: ball (centre-origin pitch m) → image x by interpolating the 65
    # annotated pitch keypoints (the panorama is strongly curved; the released homography
    # did not land on these pixels); frames without a ball keep the last centre; ±2 s
    # moving average so the window pans smoothly.
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    kp_pitch = np.array([[float(v) for v in k.strip("()").split(",")] for k in keypoints])
    kp_img = np.array(list(keypoints.values()), dtype=float)
    lin, near = LinearNDInterpolator(kp_pitch, kp_img[:, 0]), NearestNDInterpolator(kp_pitch, kp_img[:, 0])
    finite = np.isfinite(ball["x"]) & np.isfinite(ball["y"])
    bx = np.where(finite, ball["x"], 0.0) + 52.5
    by = np.where(finite, ball["y"], 0.0) + 34.0
    img_x = lin(bx, by)
    img_x = np.where(np.isnan(img_x), near(bx, by), img_x)
    cx = np.full(n + 1, np.nan)
    for f, x, good in zip(ball["frame"], img_x, (ball["status"] > 0) & finite):
        if good and 0 <= f <= n and np.isfinite(x):
            cx[int(f)] = x
    last = fw / 2.0  # forward-fill frames without a ball (start: frame centre)
    for i in range(n + 1):
        if np.isnan(cx[i]):
            cx[i] = last
        last = cx[i]
    k = 101  # 4 s at 25 fps
    cx = np.convolve(np.pad(cx, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    # Vertical band = the pitch (keypoint rows + margin): drops the stands and the
    # foreground lettering; 16:9 window of that height → players ~2.5x larger at 796x448.
    y_top = max(0, int(kp_img[:, 1].min()) - 60)
    y_bot = min(fh, int(kp_img[:, 1].max()) + 40)
    crop_h = y_bot - y_top
    crop_w = min(fw, int(round(crop_h * 16 / 9)))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"/tmp/{name}.mp4")
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    i = 0
    while True:
        good, frame = cap.read()
        if not good:
            break
        x0 = int(np.clip(cx[min(i, n)] - crop_w / 2, 0, fw - crop_w))
        y0 = y_top
        enc.stdin.write(cv2.resize(frame[y0 : y0 + crop_h, x0 : x0 + crop_w], (W, H), interpolation=cv2.INTER_AREA).tobytes())
        i += 1
    enc.stdin.close()
    enc.wait()
    subprocess.run(["cp", str(tmp), str(out_mp4)], check=True)
    hnum = "1" if half == "1st" else "2"
    ann = [{"gameTime": a["gameTime"], "label": a["label"], "position": str(a["position"]), "team": a.get("team"),
            "visibility": "visible"} for a in events["actions"] if str(a["gameTime"]).startswith(f"{hnum} ")]
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "Labels-ball.json").write_text(json.dumps({"annotations": ann}))
    meta = {"half": name, "frames": i, "events": len(ann), "split": MATCHES[match]}
    (lab_dir / "meta.json").write_text(json.dumps(meta))
    vol.commit()
    return meta


def _frames_from_crops(names: list[str], frame_root: Path) -> dict[str, int]:
    """Decode the 796x448 crop videos to T-DEED's frame{i}.jpg layout on local disk."""
    from concurrent.futures import ThreadPoolExecutor

    def one(name):
        d = frame_root / name
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"/mnt/data/st/crops/{name}.mp4", "-q:v", "4",
                        "-start_number", "0", str(d / "frame%d.jpg")], check=True)
        return name, len(list(d.glob("*.jpg")))

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, names))


def _setup_repo(train: list[str], val: list[str], counts: dict[str, int], frame_root: Path) -> None:
    """Point T-DEED's soccernetball split files / label path at the SoccerTrack halves."""
    base = Path(REPO) / "data" / "soccernetball"
    (base / "labels_path.txt").write_text("/mnt/data/st/labels")
    for split, names in (("train", train), ("val", val), ("test", val)):
        (base / f"{split}.json").write_text(json.dumps([{"video": n, "num_frames": counts[n]} for n in names]))
    cfg = json.loads((Path(REPO) / "config/SoccerNetBall/SoccerNetBall_challenge2.json").read_text())
    cfg.update(frame_dir=str(frame_root), save_dir="/mnt/data/st/runs", store_dir="/mnt/data/st/store", store_mode="store",
               epoch_num_frames=200000, num_epochs=12, warm_up_epochs=1, learning_rate=0.0004, criterion="loss",
               only_test=False, start_val_epoch=999, num_workers=8)
    cfg.pop("pretrain", None)
    d = Path(REPO) / "config" / "SoccerTrack"
    d.mkdir(exist_ok=True)
    (d / "SoccerTrack_ft.json").write_text(json.dumps(cfg, indent=1))


@app.function(image=train_image, gpu="L40S", volumes={"/mnt": vol}, cpu=8, memory=65536,  # 24 GB A10G OOMs at batch 2
              timeout=24 * 3600, ephemeral_disk=512 * 1024)
def train() -> dict:
    import sys
    import types

    import numpy as np
    import torch

    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    metas = [json.loads(p.read_text()) for p in Path("/mnt/data/st/labels").glob("*/meta.json")]
    train_names = sorted(m["half"] for m in metas if m["split"] == "train")
    val_names = sorted(m["half"] for m in metas if m["split"] == "heldout")
    frame_root = Path("/tmp/frames")
    counts = _frames_from_crops(train_names + val_names, frame_root)
    _setup_repo(train_names, val_names, counts, frame_root)

    from dataset.datasets import get_datasets
    from model.model import TDEEDModel
    from torch.utils.data import DataLoader
    from util.io import load_json

    import train_tdeed as T

    args = T.update_args(types.SimpleNamespace(model="SoccerTrack_ft", seed=1, acc_grad_iter=1),
                         load_json("config/SoccerTrack/SoccerTrack_ft.json"))
    args.crop_dim = None
    get_datasets(args)  # store mode: writes clip indexes, then reload
    args.store_mode = "load"
    classes, _, train_data, val_data, _ = get_datasets(args)
    model = TDEEDModel(args=args)
    run = Path("/mnt/data/st/runs")
    run.mkdir(parents=True, exist_ok=True)
    state = run / "last.pt"
    optimizer, scaler = model.get_optimizer({"lr": args.learning_rate})
    start, history = 0, []
    if state.exists():  # resume
        ck = torch.load(state, map_location="cpu")
        model._model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start, history = ck["epoch"] + 1, ck["history"]
    else:  # fine-tune from the SoccerNet ball checkpoint (heads that do not match are re-initialised)
        pre = torch.load("/mnt/data/tdeed/SoccerNetBall_challenge2/checkpoint_best.pt", map_location="cpu")
        own = model._model.state_dict()
        keep = {k: v for k, v in pre.items() if k in own and own[k].shape == v.shape}
        model._model.load_state_dict(keep, strict=False)
        print(f"loaded {len(keep)}/{len(own)} tensors from the pretrained checkpoint")
    loader = lambda d: DataLoader(d, shuffle=False, batch_size=args.batch_size, pin_memory=True,  # noqa: E731
                                  num_workers=args.num_workers, prefetch_factor=2)
    train_loader, val_loader = loader(train_data), loader(val_data)
    _, sched = T.get_lr_scheduler(args, optimizer, len(train_loader))
    for _ in range(start * len(train_loader)):
        sched.step()
    best = min((h["val"] for h in history), default=float("inf"))
    for epoch in range(start, args.num_epochs):
        tl = model.epoch(train_loader, optimizer, scaler, lr_scheduler=sched)
        vl = model.epoch(val_loader)
        history.append({"epoch": epoch, "train": float(tl), "val": float(vl)})
        print(history[-1], flush=True)
        if vl < best:
            best = vl
            torch.save(model._model.state_dict(), run / "best.pt")
        torch.save({"model": model._model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "history": history}, state)
        (run / "history.json").write_text(json.dumps(history, indent=1))
        vol.commit()
    return {"history": history, "best_val": best}


@app.function(image=train_image, gpu="A10G", volumes={"/mnt": vol}, timeout=6 * 3600)
def evaluate(halves: list[str], weights: str = "/mnt/data/st/runs/best.pt", threshold: float = 0.05) -> dict:
    """T-DEED inference on held-out crop videos → SoccerTrack scorer (same as the heuristic benchmark)."""
    import sys
    import types

    import torch

    os.environ["WANDB_MODE"] = "disabled"
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, "/root")
    from dataset.frame import ActionSpotInferenceDataset
    from model.model import TDEEDModel
    from torch.utils.data import DataLoader
    from util.dataset import load_classes
    from util.eval import inference
    from util.io import load_json

    import train_tdeed as T

    _setup_repo([], [], {}, Path("/tmp"))
    args = T.update_args(types.SimpleNamespace(model="SoccerTrack_ft", seed=1, acc_grad_iter=1),
                         load_json("config/SoccerTrack/SoccerTrack_ft.json"))
    args.crop_dim = None  # config -1 = no crop (train() does the same); -1 reached view() as a size
    classes = load_classes("data/soccernetball/class.txt")
    model = TDEEDModel(args=args)
    if weights.endswith("checkpoint_best.pt"):  # pretrained baseline: head may not match
        own = model._model.state_dict()
        pre = torch.load(weights, map_location="cpu")
        model._model.load_state_dict({k: v for k, v in pre.items() if k in own and own[k].shape == v.shape}, strict=False)
    else:
        ck = torch.load(weights, map_location="cpu")
        model._model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)  # last.pt = resume ckpt
    from courtlens_scripts import bench_soccertrack as BS

    label_map = {"PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "DRIVE": "dribble", "SHOT": "shot", "GOAL": "goal"}
    results = {}
    for name in halves:
        ds = ActionSpotInferenceDataset(f"/mnt/data/st/crops/{name}.mp4", clip_len=args.clip_len,
                                        overlap_len=args.clip_len // 4 * 3, stride=2, dataset="soccernetball", size=(W, H))
        out = Path("inference_output/results_inference.json")
        out.unlink(missing_ok=True)
        inference(model, DataLoader(ds, batch_size=4, num_workers=4), classes, threshold=threshold)
        raw = json.loads(out.read_text())["predictions"]
        match, half = name.split("_")
        truth = BS.load_truth(Path(f"/mnt/data/st/bas/{match}_12_class_events.json"), 1 if half == "1st" else 2)
        sweep = {}
        for th in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            pred = [{"type": label_map[p["label"]], "ts_ms": int(p["frame"] * 40), "kit": None}
                    for p in raw if p["label"] in label_map and p["confidence"] >= th]
            sweep[str(th)] = {
                "full_half": BS.score(truth, pred, 1000)["by_class"],
                # same window as the heuristic baseline (bench_soccertrack: first 12 min of M2 1st half)
                "first_12_min": BS.score(truth, [p for p in pred if p["ts_ms"] <= 720000], 1000, 720000)["by_class"],
            }
        results[name] = sweep
        Path(f"/mnt/data/st/eval_{Path(weights).stem}_{name}.json").write_text(json.dumps({"raw": raw, "sweep": sweep}))
    vol.commit()
    return results


@app.local_entrypoint()
def main(step: str = "prep", weights: str = "/mnt/data/st/runs/best.pt"):
    if step == "prep":
        for meta in prep_half.starmap(HALVES):
            print(meta)
    elif step == "train":
        print(train.remote())
    elif step == "eval":
        held = ["117093_1st", "128057_1st"]
        res = evaluate.remote(held, weights)
        Path(f"data/modal_stage/tdeed_eval_{Path(weights).stem}.json").write_text(json.dumps(res, indent=1))
        for half, sweep in res.items():
            for th, r in sweep.items():
                p = r["first_12_min"]["pass"]
                f = r["full_half"]
                print(half, th, "pass(12min) P/R/F1", p["precision"], p["recall"], p["f1"],
                      "| full pass F1", f["pass"]["f1"], "drive F1", f["drive"]["f1"], "shot P/R", f["shot"]["precision"], f["shot"]["recall"])
    else:
        raise SystemExit(step)
