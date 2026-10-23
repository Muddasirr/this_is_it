"""Pick the match ball from per-frame ball candidates.

Keeping the single most confident ball per frame failed on a wide fixed camera
(SoccerTrack v2): spare balls lying still by the touchline are bigger and sharper
than the match ball, so they won nearly every frame — 2,000 real ball points, all
in one spot, and possession followed a ball that never moved.

1. Static suppression — candidates that sit in the same place for a large share
   of the processed frames are spare balls / markings, not the match ball.
2. Trajectory choice — prefer the candidate consistent with where the ball was a
   moment ago (bounded by a maximum ball speed in ball widths per second); fall
   back to the most confident one to (re-)acquire.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

STATIC_MIN_FRAME_FRAC = 0.08  # in the same cell for ≥ 8 % of processed frames …
STATIC_MIN_SPAN_S = 20.0  # … spread over ≥ 20 s ⇒ not the match ball
MAX_SPEED_WIDTHS_PER_S = 150.0  # ≈ 33 m/s for a 0.22 m ball — hardest shots
GATE_MAX_GAP_S = 1.0  # beyond this the previous position says nothing
REACQUIRE_MIN_CONF = 0.25


def _centre(c: tuple) -> tuple[float, float]:
    return (c[1] + c[3]) / 2.0, (c[2] + c[4]) / 2.0


def static_cells(cands: list[tuple], n_frames: int) -> set[tuple[int, int]]:
    """Grid cells (≈ 2 ball widths) occupied in many frames over a long span."""
    if not cands or n_frames <= 0:
        return set()
    width = float(np.median([c[3] - c[1] for c in cands])) or 10.0
    cell = 2.0 * width
    # Short runs: a spare ball is "static" if it stays for most of what we saw
    run_s = (max(c[0] for c in cands) - min(c[0] for c in cands)) / 1000.0
    min_span_s = min(STATIC_MIN_SPAN_S, 0.6 * run_s)
    frames: dict[tuple[int, int], set[int]] = defaultdict(set)
    for c in cands:
        x, y = _centre(c)
        frames[(int(x // cell), int(y // cell))].add(int(c[0]))
    out = set()
    for key, ts in frames.items():
        # Merge the 3×3 neighbourhood so a ball resting on a cell edge still counts
        near = set().union(*(frames.get((key[0] + dx, key[1] + dy), set()) for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
        if len(near) >= STATIC_MIN_FRAME_FRAC * n_frames and (max(near) - min(near)) / 1000.0 >= min_span_s:
            out.add(key)
    return out


def select_ball(cands: list[tuple], n_frames: int) -> list[tuple]:
    """cands: (ts_ms, x1, y1, x2, y2, conf) — any number per frame. Returns ≤ 1 per ts."""
    if not cands:
        return []
    width = float(np.median([c[3] - c[1] for c in cands])) or 10.0
    cell = 2.0 * width
    static = static_cells(cands, n_frames)
    by_ts: dict[int, list[tuple]] = defaultdict(list)
    for c in cands:
        x, y = _centre(c)
        if (int(x // cell), int(y // cell)) not in static:
            by_ts[int(c[0])].append(c)
    chosen: list[tuple] = []
    last = None
    for ts in sorted(by_ts):
        options = by_ts[ts]
        pick = None
        if last is not None and (ts - last[0]) / 1000.0 <= GATE_MAX_GAP_S:
            lx, ly = _centre(last)
            reach = MAX_SPEED_WIDTHS_PER_S * width * max(0.04, (ts - last[0]) / 1000.0)
            near = [(np.hypot(_centre(o)[0] - lx, _centre(o)[1] - ly), o) for o in options]
            near = [(d, o) for d, o in near if d <= reach]
            if near:
                pick = min(near, key=lambda t: t[0])[1]
        if pick is None:
            best = max(options, key=lambda o: o[5])
            if best[5] >= REACQUIRE_MIN_CONF or last is None:
                pick = best
        if pick is not None:
            chosen.append(pick)
            last = pick
    return chosen


def summary(cands: list[tuple], chosen: list[tuple], n_frames: int) -> dict[str, Any]:
    return {
        "ball_candidates": len(cands),
        "ball_static_cells": len(static_cells(cands, n_frames)),
        "ball_selected": len(chosen),
    }
