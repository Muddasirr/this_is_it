#!/usr/bin/env python3
"""Spotter input-geometry A/B on a SoccerTrack v2 panorama (offline; T-DEED runs on Kaggle).

  backend/.venv/bin/python scripts/eval/spot_geometry_ab.py \
      --video testing_videos/st2_128057_0_720.mp4 --tracks data/st2_ball/spot_ab/tracks.jsonl \
      --match 128057 --out data/st2_ball/spot_ab

The SoccerTrack T-DEED fine-tune was trained on a 16:9 window over the *pitch band* (keypoint rows
−60/+40 px, ≈666 px of the 1080) following the annotated ball (scripts/modal/tdeed_soccertrack.py).
The app crops the full frame height (action_spotting.write_model_video) → players ~1.6x smaller
than in training. One decode of the clip writes the model input (796x448, 25 fps) for:
  A_full_ours       app today: full height, x from our ball (else players), ±2 s smoothing
  B_band_ours       pitch band from the keypoints, x from our ball
  D_band_players    pitch band from the players' boxes (no calibration needed), x from our ball
  C_band_labelled   pitch band, x from the labelled ball (the training geometry; upper bound)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.action_spotting import FPS, H, W, crop_centres, pitch_band  # noqa: E402

ST2 = ROOT / "data/eval/soccertrack2"


def labelled_centres(match: str, n: int, width: int) -> np.ndarray:
    """Training crop centre: labelled ball (status > 0) → image x via the keypoints, ffill, ±2 s."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    kp = json.loads((ST2 / f"raw/{match}/{match}_keypoints.json").read_text())
    pitch = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])
    img = np.array(list(kp.values()), dtype=float)
    lin, near = LinearNDInterpolator(pitch, img[:, 0]), NearestNDInterpolator(pitch, img[:, 0])
    ball = np.load(ST2 / f"ball/{match}_1st_ball.npz")
    ok = np.isfinite(ball["x"]) & np.isfinite(ball["y"])
    bx, by = np.where(ok, ball["x"], 0.0) + 52.5, np.where(ok, ball["y"], 0.0) + 34.0
    ix = lin(bx, by)
    ix = np.where(np.isnan(ix), near(bx, by), ix)
    cx = np.full(n, np.nan)
    for f, x, good in zip(ball["frame"], ix, (ball["status"] > 0) & ok):
        if good and 1 <= f <= n and np.isfinite(x):
            cx[int(f) - 1] = x
    last = width / 2.0
    for i in range(n):
        cx[i] = last = last if np.isnan(cx[i]) else cx[i]
    k = 101
    return np.convolve(np.pad(cx, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="jsonl: {class, bbox_series} per track")
    ap.add_argument("--match", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import cv2

    dets = [SimpleNamespace(class_name=r["class"], bbox_series=r["bbox_series"])
            for r in map(json.loads, args.tracks.read_text().splitlines())]
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert abs(fps - FPS) < 0.01, "clip must be 25 fps (no resampling here)"

    kp = np.array(list(json.loads((ST2 / f"raw/{args.match}/{args.match}_keypoints.json").read_text()).values()))
    kp_band = (max(0, int(kp[:, 1].min()) - 60), min(fh, int(kp[:, 1].max()) + 40))
    pl_band = pitch_band(dets, fh)
    ours, lab = crop_centres(dets, n, fps, fw), labelled_centres(args.match, n, fw)
    variants = {
        "A_full_ours": ((0, fh), ours),
        "B_band_ours": (kp_band, ours),
        "D_band_players": (pl_band, ours),
        "C_band_labelled": (kp_band, lab),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: {"band": list(b), "crop": [min(fw, round((b[1] - b[0]) * 16 / 9)), b[1] - b[0]]} for k, (b, _) in variants.items()}
    print(json.dumps(meta))
    writers = {k: cv2.VideoWriter(str(args.out / f"{k}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
               for k in variants}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for k, ((y0, y1), cx) in variants.items():
            cw = min(fw, round((y1 - y0) * 16 / 9))
            x0 = int(np.clip(cx[min(i, len(cx) - 1)] - cw / 2, 0, fw - cw))
            writers[k].write(cv2.resize(frame[y0:y1, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA))
        i += 1
    for wr in writers.values():
        wr.release()
    (args.out / "meta.json").write_text(json.dumps({"frames": i, **meta}, indent=1))
    print(f"{i} frames → {args.out}")


if __name__ == "__main__":
    main()
