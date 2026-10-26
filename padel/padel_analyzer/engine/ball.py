"""Ball trajectory from per-frame candidates.

Per-frame argmax confidence loses to static look-alikes (balls lying by the glass, lights, logos).
Instead: link candidates into tracklets with a constant-velocity gate, score tracklets
(confidence mass x motion), drop static ones, choose the best covering tracklet per frame,
then bridge short gaps. Trajectory-level selection is the largest single lever reported for
padel ball tracking (lmrt0572: 16 -> 72 % recall).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallTrack:
    xy: np.ndarray  # (n, 2) px, nan when unknown
    conf: np.ndarray  # (n,) detector confidence (0 for interpolated / missing)
    observed: np.ndarray  # (n,) bool detected (not interpolated)


@dataclass
class _Tracklet:
    frames: list
    pts: list
    confs: list

    def predict(self, f: int) -> np.ndarray:
        p = np.asarray(self.pts[-1])
        if len(self.pts) >= 2:
            dt = self.frames[-1] - self.frames[-2]
            v = (np.asarray(self.pts[-1]) - np.asarray(self.pts[-2])) / max(dt, 1)
            return p + v * (f - self.frames[-1])
        return p


def track(cands: np.ndarray, min_conf: float = 0.35, gate_px: float = 45.0, max_speed_px: float = 160.0,
          max_gap: int = 5, max_fill: int = 4, min_len: int = 3, static_px: float = 3.0) -> BallTrack:
    """cands: (n, K, 5) xyxy conf (conf 0 = empty). min_conf / max_fill chosen cross-match on PT100
    (in-rally F1, docs/EVAL.md §3): 0.35 / 4 won on both finals."""
    n = len(cands)
    centres = np.stack([(cands[..., 0] + cands[..., 2]) / 2, (cands[..., 1] + cands[..., 3]) / 2], -1)
    conf = cands[..., 4]
    active: list[_Tracklet] = []
    done: list[_Tracklet] = []
    for f in range(n):
        ok = np.where(conf[f] >= min_conf)[0]
        still = []
        for t in active:
            if f - t.frames[-1] > max_gap:
                done.append(t)
            else:
                still.append(t)
        active = still
        used = set()
        if len(ok) and active:
            # greedy by distance to each tracklet's prediction
            pairs = []
            for ti, t in enumerate(active):
                pred = t.predict(f)
                lim = gate_px + 0.5 * np.linalg.norm(pred - np.asarray(t.pts[-1]))
                lim = min(lim, max_speed_px * (f - t.frames[-1]))
                for ci in ok:
                    d = np.linalg.norm(centres[f, ci] - pred)
                    if d <= lim:
                        pairs.append((d - 20 * conf[f, ci], ti, ci))
            took_t = set()
            for _, ti, ci in sorted(pairs):
                if ti in took_t or ci in used:
                    continue
                t = active[ti]
                t.frames.append(f)
                t.pts.append(centres[f, ci])
                t.confs.append(conf[f, ci])
                took_t.add(ti)
                used.add(ci)
        for ci in ok:
            if ci not in used:
                active.append(_Tracklet([f], [centres[f, ci]], [conf[f, ci]]))
    done.extend(active)

    # score + filter
    scored = []
    for t in done:
        if len(t.frames) < min_len:
            if not (len(t.frames) >= 1 and max(t.confs) >= 0.5):
                continue
        pts = np.asarray(t.pts)
        spread = np.linalg.norm(pts - np.median(pts, 0), axis=1).max() if len(pts) > 1 else 0.0
        if len(t.frames) >= 8 and spread < static_px * 2:
            continue  # static look-alike
        score = float(np.sum(t.confs)) * (1.0 if spread >= static_px else 0.3)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])

    xy = np.full((n, 2), np.nan)
    cf = np.zeros(n)
    owner = np.full(n, -1)
    for k, (score, t) in enumerate(scored):
        fr = np.asarray(t.frames)
        span = slice(fr[0], fr[-1] + 1)
        # a tracklet only claims frames not already owned by a better one
        if (owner[span] >= 0).mean() > 0.5:
            continue
        for f, p, c in zip(t.frames, t.pts, t.confs):
            if owner[f] < 0:
                xy[f], cf[f], owner[f] = p, c, k
    observed = ~np.isnan(xy[:, 0])
    xy = _fill(xy, max_fill, max_speed_px)
    return BallTrack(xy, cf, observed)


def track_for_events(cands: np.ndarray) -> BallTrack:
    """Recall-oriented track for hit features: a hit needs the trajectory on both sides of the contact,
    so low-confidence candidates and longer gap fills help even though they cost display precision."""
    return track(cands, min_conf=0.08, max_fill=8)


def _fill(xy: np.ndarray, max_fill: int, max_speed_px: float) -> np.ndarray:
    out = xy.copy()
    obs = np.where(~np.isnan(xy[:, 0]))[0]
    for a, b in zip(obs[:-1], obs[1:]):
        gap = b - a
        if 1 < gap <= max_fill and np.linalg.norm(xy[b] - xy[a]) <= max_speed_px * gap:
            t = (np.arange(a + 1, b) - a) / gap
            out[a + 1:b] = xy[a] + t[:, None] * (xy[b] - xy[a])
    return out


def velocity(xy: np.ndarray, k: int = 2) -> np.ndarray:
    """Central difference over +-k frames (px/frame), nan where undefined."""
    v = np.full_like(xy, np.nan)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) / (2 * k)
    return v
