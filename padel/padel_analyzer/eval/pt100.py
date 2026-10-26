"""PadelTracker100 ground truth as dense per-frame arrays (cached to .npz after the first load)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHES = {"F": "2022_BCN_FinalF_1", "M": "2022_BCN_FinalM_1"}
SHOT_CLASSES = ["serve", "forehand", "backhand", "smash", "other"]
_SHOT_MAP = {"Serve": "serve", "Forehand": "forehand", "Backhand": "backhand", "Smash": "smash",
             "Other": "other", "Dropshot": "other"}
MAXP = 5


def default_label_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "pt100" / "labels"


@dataclass
class Shot:
    frame: int  # window centre ~ racket contact
    start: int
    end: int
    kind: str


@dataclass
class MatchGT:
    key: str
    n: int
    H: np.ndarray  # px -> m
    ball: np.ndarray  # (n, 2) px centre, nan when no ball
    pbox: np.ndarray  # (n, MAXP, 4) xyxy, nan padded
    pkpt: np.ndarray  # (n, MAXP, 17, 3)
    pos_m: np.ndarray  # (n, MAXP, 2) floor metres, nan padded
    shots: list[Shot]
    shot_range: tuple[int, int]  # frames covered by the shot labels [a, b)

    def tracks_m(self) -> np.ndarray:
        """(n, 4, 2) GT positions linked into 4 continuous tracks (2 slots per net side)."""
        return link_tracks(self.pos_m)


def link_tracks(pos: np.ndarray, max_step_m: float = 1.0) -> np.ndarray:
    """Link per-frame positions into 4 slots. Slots are re-seeded (far pair, near pair, each
    left->right) whenever an assignment needs a jump > max_step_m (cut / side change)."""
    n = len(pos)
    out = np.full((n, 4, 2), np.nan)
    prev = None
    for f in range(n):
        p = pos[f][~np.isnan(pos[f, :, 0])]
        if len(p) > 4:
            p = p[np.argsort(np.abs(p[:, 0] - 5) + np.abs(p[:, 1] - 10))][:4]
        if prev is not None and len(p):
            cost = np.linalg.norm(prev[:, None] - p[None], axis=2)
            r, c = linear_sum_assignment(cost)
            if cost[r, c].max() <= max_step_m:
                out[f, r] = p[c]
                prev[r] = p[c]
                continue
        if len(p) == 4:
            p = p[np.argsort(p[:, 1])]
            far, near = p[:2][np.argsort(p[:2, 0])], p[2:][np.argsort(p[2:, 0])]
            out[f] = np.vstack([far, near])
            prev = out[f].copy()
    return out


def path_length(tracks: np.ndarray, max_step_m: float = 0.5) -> np.ndarray:
    """Distance per slot summing only plausible frame-to-frame steps (< 15 m/s at 30 fps)."""
    d = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    d = np.where(np.isnan(d) | (d > max_step_m), 0.0, d)
    return d.sum(0)


def _frame(name: str) -> int:
    return int(name.split("_")[1].split(".")[0])


def load(key: str, label_dir: Path | None = None) -> MatchGT:
    label_dir = label_dir or default_label_dir()
    m = MATCHES[key]
    cache = label_dir / f"{m}_gt_cache.npz"
    shots_csv = label_dir / f"{m}_shots.csv"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return MatchGT(key, int(z["n"]), z["H"], z["ball"], z["pbox"], z["pkpt"], z["pos_m"],
                       [Shot(*s) for s in z["shots"].tolist()], tuple(z["shot_range"]))
    ball = json.loads((label_dir / f"{m}_ball.json").read_text())
    n = len(ball["images"])
    img2f = {im["id"]: _frame(im["file_name"]) for im in ball["images"]}
    b = np.full((n, 2), np.nan)
    for a in ball["annotations"]:
        if a["category_id"] == 1:
            x, y, w, h = a["bbox"]
            b[img2f[a["image_id"]]] = (x + w / 2, y + h / 2)
    del ball
    pose = json.loads((label_dir / f"{m}_pose.json").read_text())
    pimg2f = {im["id"]: _frame(im["file_name"]) for im in pose["images"]}
    pbox = np.full((n, MAXP, 4), np.nan)
    pkpt = np.zeros((n, MAXP, 17, 3), np.float32)
    slot = np.zeros(n, int)
    ann_slot: dict[int, tuple[int, int]] = {}
    for a in pose["annotations"]:
        f = pimg2f[a["image_id"]]
        if f >= n or slot[f] >= MAXP:
            continue
        k = slot[f]
        x, y, w, h = a["bbox"]
        pbox[f, k] = (x, y, x + w, y + h)
        pkpt[f, k] = np.asarray(a["keypoints"], np.float32).reshape(17, 3)
        ann_slot[a["id"]] = (f, k)
        slot[f] += 1
    del pose
    hom = json.loads((label_dir / f"{m}_homography.json").read_text())["homography"]
    H = np.asarray(hom[0]["H"], np.float64)
    pos = np.full((n, MAXP, 2), np.nan)
    for e in hom:
        for aid, xy in e["location_m"].items():
            fk = ann_slot.get(int(aid))
            if fk:
                pos[fk] = xy
    del hom
    shots, rng = _load_shots(shots_csv)
    np.savez_compressed(cache, n=n, H=H, ball=b, pbox=pbox, pkpt=pkpt, pos_m=pos,
                        shots=np.array([(s.frame, s.start, s.end, s.kind) for s in shots], dtype=object),
                        shot_range=np.array(rng))
    return MatchGT(key, n, H, b, pbox, pkpt, pos, shots, rng)


def _load_shots(path: Path) -> tuple[list[Shot], tuple[int, int]]:
    rows = list(csv.reader(path.open(), delimiter=";"))[1:]
    frames = [_frame(r[0]) for r in rows]
    shots: list[Shot] = []
    cur = None
    for f, r in zip(frames, rows):
        lab = r[2] if r[1] == "1" else None
        if lab and cur and cur[2] == lab and f == cur[1] + 1:
            cur[1] = f
            continue
        if cur:
            shots.append(Shot((cur[0] + cur[1]) // 2, cur[0], cur[1], _SHOT_MAP[cur[2]]))
            cur = None
        if lab:
            cur = [f, f, lab]
    if cur:
        shots.append(Shot((cur[0] + cur[1]) // 2, cur[0], cur[1], _SHOT_MAP[cur[2]]))
    return shots, (min(frames), max(frames) + 1)
