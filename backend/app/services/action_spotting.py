"""Learned ball-action spotting (roadmap W7): T-DEED fine-tuned on SoccerTrack v2.

Held-out SoccerTrack halves (M2, M7), annotated-ball crop: pass F1 0.80–0.82 vs 0.40
for the possession heuristics; dribble F1 0.72–0.77; shots P 0.50 / R 0.60–0.80.

Runs where T-DEED (GPL-3.0, server-side only) and the weights are present — the Modal
GPU image (TDEED_REPO, TDEED_WEIGHTS). Input to the model: 25 fps, 796x448. 16:9-ish
cameras are resized whole; panoramas (aspect > 2.2) get the training crop: a 16:9 window
over the frame height that follows the play (our own ball, else the players' centroid,
smoothed ±2 s) — the same geometry the model was trained on.

The model says *what* and *when*; `app.services.events.events_from_spotting` adds *who*
from the possession timeline.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

W, H, FPS = 796, 448, 25
PANORAMA_ASPECT = 2.2
LABEL_TO_EVENT = {"PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "DRIVE": "dribble", "SHOT": "shot",
                  "GOAL": "goal"}
# Per-class confidence. Football (SoccerTrack fine-tune): the held-out sweep, best F1 on M2 + M7.
# Futsal fine-tune: TDEED_THRESHOLDS (docker-compose; from the labelled futsal test window).
FOOTBALL_THRESHOLDS = {"pass": 0.2, "dribble": 0.3, "shot": 0.3, "goal": 0.3}
THRESHOLDS = {**FOOTBALL_THRESHOLDS, **json.loads(os.environ.get("TDEED_THRESHOLDS") or "{}")}


def is_football(fmt: str | None = None) -> bool:
    from app.services.court_format import get_court_format

    return (fmt or get_court_format()) == "11v11"


def thresholds(fmt: str | None = None) -> dict[str, float]:
    """The thresholds of the model this format runs: football → SoccerTrack sweep, futsal → env."""
    return dict(FOOTBALL_THRESHOLDS) if is_football(fmt) else dict(THRESHOLDS)


def weights_path() -> str | None:
    """Football matches use the SoccerTrack model (TDEED_WEIGHTS_FOOTBALL), futsal the futsal one."""
    if is_football() and os.environ.get("TDEED_WEIGHTS_FOOTBALL"):
        return os.environ["TDEED_WEIGHTS_FOOTBALL"]
    return os.environ.get("TDEED_WEIGHTS")
# T-DEED SoccerTrack fine-tune config (scripts/modal/tdeed_soccertrack.py) — inference needs only these
CONFIG = {"clip_len": 100, "feature_arch": "rny008_gsf", "temporal_arch": "ed_sgp_mixer", "n_layers": 2,
          "sgp_ks": 9, "sgp_r": 4, "num_classes": 12, "radi_displacement": 4, "modality": "rgb", "crop_dim": None,
          "dataset": "soccernetball", "batch_size": 4, "num_workers": 0, "pretrain": None}


def available(device: str | None = None) -> bool:
    """Repo + weights present, and a GPU (CPU runs ~7x slower than real time: a 40-min match takes
    hours — the heuristics are used there unless TDEED_ALLOW_CPU=1)."""
    repo, weights = os.environ.get("TDEED_REPO"), weights_path()
    if not (repo and weights and Path(repo).is_dir() and Path(weights).is_file()):
        return False
    on_cpu = str(device or "").lower() in {"", "cpu"}
    return not on_cpu or os.environ.get("TDEED_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes"}


def crop_centres(detections: list, n_frames: int, fps: float, width: int) -> np.ndarray:
    """Per source frame x-centre of the play: ball where seen, else players' centroid; ±2 s smoothing."""
    xs = np.full(n_frames, np.nan)
    ball_seen = np.zeros(n_frames, bool)
    persons: dict[int, list[float]] = {}
    for d in detections:
        for p in d.bbox_series or []:
            if "x1" not in p:
                continue
            f = int(round(int(p["ts_ms"]) * fps / 1000.0))
            if not 0 <= f < n_frames:
                continue
            cx = (float(p["x1"]) + float(p["x2"])) / 2
            if d.class_name == "sports_ball" and not p.get("interpolated"):
                xs[f], ball_seen[f] = cx, True
            elif d.class_name == "person":
                persons.setdefault(f, []).append(cx)
    for f, cxs in persons.items():
        if not ball_seen[f]:
            xs[f] = float(np.median(cxs))
    last = width / 2.0
    for i in range(n_frames):
        if np.isnan(xs[i]):
            xs[i] = last
        last = xs[i]
    k = int(4 * fps) | 1
    return np.convolve(np.pad(xs, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def write_model_video(video_path: str, detections: list, out_path: str, until_ms: int | None = None) -> dict:
    """Source video → 25 fps 796x448 H.264 in the training geometry (see module doc), up to until_ms."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panorama = fw / max(1, fh) > PANORAMA_ASPECT
    cx = crop_centres(detections, max(1, n), fps, fw) if panorama else None
    crop_w = min(fw, int(round(fh * 16 / 9))) if panorama else fw
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    enc = subprocess.Popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                            "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    step = fps / FPS  # resample to 25 fps by nearest source frame
    i, next_src, written = 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (until_ms is not None and i * 1000.0 / fps > until_ms):
            break
        while i >= next_src - 1e-6:
            if panorama:
                x0 = int(np.clip(cx[min(i, len(cx) - 1)] - crop_w / 2, 0, fw - crop_w))
                view = frame[:, x0 : x0 + crop_w]
            else:
                view = frame
            enc.stdin.write(cv2.resize(view, (W, H), interpolation=cv2.INTER_AREA).tobytes())
            written += 1
            next_src += step
        i += 1
    enc.stdin.close()
    enc.wait()
    cap.release()
    return {"frames": written, "panorama": panorama, "source_fps": fps}


def _model(device: str):
    repo = os.environ["TDEED_REPO"]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from model.model import TDEEDModel

    args = types.SimpleNamespace(**CONFIG)
    model = TDEEDModel(device=device, args=args)
    ck = torch.load(weights_path(), map_location="cpu", weights_only=False)
    model._model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    return model


def spot(video_path: str, detections: list, device: str = "cuda") -> list[dict[str, Any]]:
    """[{ts_ms, label, event_type, confidence}] for the video (camera time), or [] if unavailable."""
    if not available(device):
        return []
    repo = os.environ["TDEED_REPO"]
    cwd = os.getcwd()
    try:
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmp:
            mp4 = str(Path(tmp) / "model_input.mp4")
            last = max((int(p["ts_ms"]) for d in detections for p in (d.bbox_series or []) if "ts_ms" in p), default=None)
            meta = write_model_video(video_path, detections, mp4, until_ms=last)  # only what detection covered
            os.chdir(repo)  # T-DEED resolves data/<dataset>/class.txt relative to its repo
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from dataset.frame import ActionSpotInferenceDataset
            from util.dataset import load_classes
            from util.eval import inference

            classes = load_classes(os.path.join("data", "soccernetball", "class.txt"))
            model = _model(device)
            ds = ActionSpotInferenceDataset(mp4, clip_len=CONFIG["clip_len"], overlap_len=CONFIG["clip_len"] // 4 * 3,
                                            stride=2, dataset="soccernetball", size=(W, H))
            out = Path("inference_output/results_inference.json")
            out.unlink(missing_ok=True)
            # num_workers=0: the dataset is iterable — every worker would replay the whole video.
            # Batch: 4 clips only on ≥ 20 GB GPUs; the 16 GB Kaggle T4 (detector / ReID / SigLIP
            # still resident) and the CPU worker both ran out of memory at 4 → one clip at a time.
            import torch

            batch = 1
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                if torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3:
                    batch = CONFIG["batch_size"]
            try:
                inference(model, DataLoader(ds, batch_size=batch, num_workers=0), classes, threshold=0.05)
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                torch.cuda.empty_cache()
                logger.warning("Spotting OOM at batch %d — retrying one clip at a time", batch)
                inference(model, DataLoader(ds, batch_size=1, num_workers=0), classes, threshold=0.05)
            raw = json.loads(out.read_text())["predictions"]
    except Exception as exc:  # noqa: BLE001 — never fail detection for the spotter
        logger.warning("Action spotting failed (%s) — heuristic events will be used", exc)
        return []
    finally:
        os.chdir(cwd)
    spotted = []  # everything ≥ 0.05 is returned; events_from_spotting applies THRESHOLDS (tunable offline)
    for p in raw:
        ev = LABEL_TO_EVENT.get(p["label"])
        if ev:
            spotted.append({"ts_ms": int(round(p["frame"] * 1000 / FPS)), "label": p["label"], "event_type": ev,
                            "confidence": round(float(p["confidence"]), 3)})
    logger.info("T-DEED spotted %d events (%s)", len(spotted), meta)
    return spotted
