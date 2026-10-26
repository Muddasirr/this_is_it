"""Racket hits: per-frame features over the ball + player tracks -> classifier -> peaks.

Features combine what a hit looks like in 2D: the ball's image velocity changes abruptly, a
player's wrist is next to the ball, and that wrist moves fast. A gradient-boosted classifier
scores every frame; peak picking with non-maximum suppression gives hit frames; the hitter is the
player whose wrist is closest to the ball at the peak.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ball import BallTrack
from .players import PlayerTracks

WRISTS = (9, 10)
FEATURES = [
    "obs_frac", "has_ball", "cos_turn3", "cos_turn6", "speed_b3", "speed_a3", "speed_ratio3", "acc3", "acc6",
    "vy_flip", "d_wrist", "d_wrist_m3", "d_wrist_p3", "d_wrist_2nd", "in_box", "rel_x", "rel_y", "wrist_speed",
    "wrist_above", "acc3_m2", "acc3_p2", "hitter_side",
]
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass
class Hit:
    frame: int
    slot: int  # player slot at that frame
    score: float


def _scale(tracks: PlayerTracks) -> np.ndarray:
    """(n, 4) player box height in px (nan when unknown), forward/back filled."""
    h = tracks.box[:, :, 3] - tracks.box[:, :, 1]
    out = h.copy()
    for s in range(4):
        x = h[:, s]
        ok = ~np.isnan(x)
        if ok.any():
            idx = np.where(ok, np.arange(len(x)), 0)
            np.maximum.accumulate(idx, out=idx)
            out[:, s] = np.where(ok.cumsum() > 0, x[idx], np.nanmedian(x))
    return out


def _wrists(tracks: PlayerTracks) -> np.ndarray:
    """(n, 4, 2, 2) wrist px, nan when low confidence."""
    w = tracks.kpt[:, :, WRISTS, :2].astype(float)
    c = tracks.kpt[:, :, WRISTS, 2]
    w[c < 0.3] = np.nan
    return w


def features(ball: BallTrack, tracks: PlayerTracks) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X (n, len(FEATURES)), nearest_slot (n,))."""
    n = len(ball.xy)
    xy = ball.xy
    obs = ball.observed.astype(float)
    X = np.full((n, len(FEATURES)), np.nan, np.float32)
    col = {k: i for i, k in enumerate(FEATURES)}
    ker = np.ones(13) / 13
    X[:, col["obs_frac"]] = np.convolve(obs, ker, "same")
    X[:, col["has_ball"]] = ~np.isnan(xy[:, 0])

    def shift(a, k):
        out = np.full_like(a, np.nan)
        if k > 0:
            out[:-k] = a[k:]
        elif k < 0:
            out[-k:] = a[:k]
        else:
            out[:] = a
        return out

    scale = _scale(tracks)
    near_scale = np.nanmedian(scale, axis=1)
    near_scale = np.where(np.isnan(near_scale), 100.0, near_scale)
    for k in (3, 6):
        vb = (xy - shift(xy, -k)) / k
        va = (shift(xy, k) - xy) / k
        nb, na = np.linalg.norm(vb, axis=1), np.linalg.norm(va, axis=1)
        cos = (vb * va).sum(1) / (nb * na + 1e-6)
        acc = np.linalg.norm(va - vb, axis=1) / near_scale
        X[:, col[f"cos_turn{k}"]] = cos
        X[:, col[f"acc{k}"]] = acc
        if k == 3:
            X[:, col["speed_b3"]] = nb / near_scale
            X[:, col["speed_a3"]] = na / near_scale
            X[:, col["speed_ratio3"]] = np.log((na + 1) / (nb + 1))
            X[:, col["vy_flip"]] = np.sign(vb[:, 1]) * -np.sign(va[:, 1])
            X[:, col["acc3_m2"]] = shift(acc, -2)
            X[:, col["acc3_p2"]] = shift(acc, 2)

    wr = _wrists(tracks)
    d = np.linalg.norm(wr - xy[:, None, None, :], axis=3)  # (n, 4, 2)
    d = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2) / scale  # (n, 4)
    d = np.where(np.isfinite(d), d, np.nan)
    dd = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(dd, axis=1)
    rows = np.arange(n)
    dmin = dd[rows, nearest]
    srt = np.sort(dd, axis=1)
    X[:, col["d_wrist"]] = np.where(np.isfinite(dmin), dmin, np.nan)
    X[:, col["d_wrist_m3"]] = shift(X[:, col["d_wrist"]], -3)
    X[:, col["d_wrist_p3"]] = shift(X[:, col["d_wrist"]], 3)
    X[:, col["d_wrist_2nd"]] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
    bx = tracks.box[rows, nearest]
    h = scale[rows, nearest]
    X[:, col["in_box"]] = ((xy[:, 0] >= bx[:, 0]) & (xy[:, 0] <= bx[:, 2]) & (xy[:, 1] >= bx[:, 1] - 0.3 * h)
                           & (xy[:, 1] <= bx[:, 3]))
    X[:, col["rel_x"]] = (xy[:, 0] - (bx[:, 0] + bx[:, 2]) / 2) / h
    X[:, col["rel_y"]] = (xy[:, 1] - bx[:, 1]) / h
    wn = wr[rows, nearest]  # (n, 2, 2)
    wspeed = np.linalg.norm(shift(wn, 2) - shift(wn, -2), axis=2) / 4
    X[:, col["wrist_speed"]] = np.nanmax(np.where(np.isnan(wspeed), -1, wspeed), axis=1) / h
    sh_y = np.nanmin(tracks.kpt[rows, nearest][:, [5, 6], 1], axis=1)
    X[:, col["wrist_above"]] = (np.nanmin(np.where(np.isnan(wn[:, :, 1]), np.inf, wn[:, :, 1]), axis=1) < sh_y)
    X[:, col["hitter_side"]] = (nearest >= 2).astype(float)
    return X, nearest


def labels(n: int, hit_frames: list[int], radius: int = 3) -> np.ndarray:
    y = np.zeros(n, np.int8)
    for f in hit_frames:
        y[max(0, f - radius):f + radius + 1] = 1
    return y


def heuristic_score(X: np.ndarray) -> np.ndarray:
    c = {k: i for i, k in enumerate(FEATURES)}
    turn = np.nan_to_num(1 - X[:, c["cos_turn3"]], nan=0) / 2
    prox = np.exp(-np.nan_to_num(X[:, c["d_wrist"]], nan=9.0) / 0.35)
    acc = np.clip(np.nan_to_num(X[:, c["acc3"]], nan=0) / 0.05, 0, 1)
    return turn * prox * acc


class HitModel:
    def __init__(self, path: Path | str | None = None):
        self.booster = None
        self.threshold = 0.5
        p = Path(path) if path else MODEL_DIR / "hit_lgbm.txt"
        if p.exists():
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(p))
            meta = p.with_suffix(".json")
            if meta.exists():
                import json

                self.threshold = json.loads(meta.read_text()).get("threshold", 0.5)
        else:
            self.threshold = 0.15

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            return heuristic_score(X)
        return self.booster.predict(X)

    def detect(self, ball: BallTrack, tracks: PlayerTracks, nms: int = 12) -> tuple[list[Hit], np.ndarray]:
        X, nearest = features(ball, tracks)
        s = self.score(X)
        return pick_peaks(s, nearest, self.threshold, nms), s


def pick_peaks(score: np.ndarray, nearest: np.ndarray, thr: float, nms: int = 12) -> list[Hit]:
    sm = np.convolve(score, np.ones(3) / 3, "same")
    cand = np.where(sm >= thr)[0]
    hits: list[Hit] = []
    for f in cand[np.argsort(-sm[cand])]:
        if any(abs(f - h.frame) < nms for h in hits):
            continue
        hits.append(Hit(int(f), int(nearest[f]), float(sm[f])))
    return sorted(hits, key=lambda h: h.frame)
