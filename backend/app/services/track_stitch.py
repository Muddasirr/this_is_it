"""Offline tracklet stitching (roadmap W3).

Online trackers (BoT-SORT) split one player into many short tracks whenever he is
occluded, leaves the frame or the detector misses him for longer than the track
buffer. Post-match we can look at the whole video: link a track that *ends* to a
track that *starts* shortly after when the motion, appearance and team agree —
the AFLink idea from StrongSORT (Du et al., 2023), done with explicit gates
instead of a learned link model.

Cost of linking a → b (lower is better), only for 0 < gap ≤ max_gap_ms:
  motion  = |a's end centre extrapolated over the gap − b's start centre| / player height
  appear  = 1 − cos(ReID_a, ReID_b)          (when both have embeddings)
Hard gates: motion ≤ max_motion(gap), appear ≤ max_appear, same team when both known,
and the two chains must never be on screen at the same time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_GAP_MS = 4000
# Player heights a player can plausibly move: a base for detector jitter plus a
# running-speed allowance per second of gap (futsal sprint ≈ 4 heights/s on screen).
MOTION_BASE_H = 1.0
MOTION_PER_S_H = 2.5
MAX_APPEAR_DIST = 0.35
APPEAR_WEIGHT = 2.0
VEL_WINDOW = 5  # observations used for the end/start velocity


def _centre_h(box) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2, max(1.0, y2 - y1)  # foot point: stable under occlusion of the torso


def _ends(obs: list[tuple[int, list[float]]]):
    """(start_t, start_xy, end_t, end_xy, end_velocity per ms, median height)."""
    pts = [(t, *_centre_h(b)) for t, b in obs]
    t0, x0, y0, _ = pts[0]
    t1, x1, y1, _ = pts[-1]
    tail = pts[-VEL_WINDOW:]
    if len(tail) >= 2 and tail[-1][0] > tail[0][0]:
        dt = tail[-1][0] - tail[0][0]
        vel = ((tail[-1][1] - tail[0][1]) / dt, (tail[-1][2] - tail[0][2]) / dt)
    else:
        vel = (0.0, 0.0)
    h = float(np.median([p[3] for p in pts]))
    return t0, (x0, y0), t1, (x1, y1), vel, h


def _cos_dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(1.0 - np.dot(a, b) / (na * nb))


def link_cost(ea, eb, emb_a=None, emb_b=None, team_a=None, team_b=None,
              max_gap_ms: int = MAX_GAP_MS) -> float | None:
    """None when a → b is not allowed; else the link cost."""
    _, _, a_end_t, a_end_xy, a_vel, a_h = ea
    b_start_t, b_start_xy, _, _, _, b_h = eb
    gap = b_start_t - a_end_t
    if gap <= 0 or gap > max_gap_ms:
        return None
    if team_a and team_b and team_a != team_b:
        return None
    # Extrapolate at most 1 s of motion: velocity from a few noisy boxes is unreliable beyond that
    horizon = min(gap, 1000)
    px, py = a_end_xy[0] + a_vel[0] * horizon, a_end_xy[1] + a_vel[1] * horizon
    h = (a_h + b_h) / 2.0
    motion = float(np.hypot(px - b_start_xy[0], py - b_start_xy[1]) / h)
    if motion > MOTION_BASE_H + MOTION_PER_S_H * gap / 1000.0:
        return None
    appear = _cos_dist(emb_a, emb_b)
    if appear is not None and appear > MAX_APPEAR_DIST:
        return None
    return motion + APPEAR_WEIGHT * (appear if appear is not None else MAX_APPEAR_DIST / 2)


def stitch(
    tracks: dict[Any, list[tuple[int, list[float]]]],
    emb: dict[Any, Any] | None = None,
    team: dict[Any, str | None] | None = None,
    max_gap_ms: int = MAX_GAP_MS,
) -> dict[Any, Any]:
    """Map every track id → the id of its chain head (unchanged ids map to themselves).

    `tracks`: id → time-sorted [(t_ms, [x1, y1, x2, y2]), ...].
    Greedy on ascending cost; one successor / predecessor per track; chains that
    would be visible at the same time are never joined.
    """
    emb = emb or {}
    team = team or {}
    ids = [k for k, v in tracks.items() if v]
    ends = {k: _ends(tracks[k]) for k in ids}
    by_start = sorted(ids, key=lambda k: ends[k][0])
    starts = [ends[k][0] for k in by_start]
    cand = []
    import bisect

    for a in ids:
        a_end = ends[a][2]
        lo = bisect.bisect_right(starts, a_end)
        hi = bisect.bisect_right(starts, a_end + max_gap_ms)
        for b in by_start[lo:hi]:
            if b == a:
                continue
            c = link_cost(ends[a], ends[b], emb.get(a), emb.get(b), team.get(a), team.get(b), max_gap_ms)
            if c is not None:
                cand.append((c, a, b))
    cand.sort(key=lambda x: x[0])

    head = {k: k for k in ids}
    spans: dict[Any, list[tuple[int, int]]] = {k: [(ends[k][0], ends[k][2])] for k in ids}
    has_succ: set = set()
    has_pred: set = set()

    def root(k):
        while head[k] != k:
            head[k] = head[head[k]]
            k = head[k]
        return k

    for _c, a, b in cand:
        if a in has_succ or b in has_pred:
            continue
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        if any(s1 < e2 and s2 < e1 for s1, e1 in spans[ra] for s2, e2 in spans[rb]):
            continue  # both chains on screen at once → two different players
        head[rb] = ra
        spans[ra].extend(spans.pop(rb))
        has_succ.add(a)
        has_pred.add(b)
    return {k: root(k) for k in ids}
