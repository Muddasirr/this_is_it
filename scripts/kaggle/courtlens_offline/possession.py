"""Vidal-Codina-style possession (image plane).

PZ around each player; gain prefers ball Δspeed/Δdirection; loss with re-entry grace.
"""

from __future__ import annotations

from typing import Any

from vision import Detection

POSSESSION_RADIUS_FRAC = 0.14
MIN_POSSESSION_SAMPLES = 3
MIN_POSSESSION_MS = 220
PERSON_TIME_WINDOW_MS = 2000
REENTRY_GRACE_MS = 400
# Contact proof (paper §4.2) — relative to frame diagonal
MIN_DIR_DOT_CHANGE = 0.35  # 1 - cos(θ) approx via 1 - max(0, dot)
MIN_SPEED_CHANGE_FRAC = 0.00004


def ball_density_per_s(ball_series: list[dict[str, Any]]) -> float:
    if len(ball_series) < 2:
        return 0.0
    t0 = int(ball_series[0].get("ts_ms", 0))
    t1 = int(ball_series[-1].get("ts_ms", 0))
    return len(ball_series) / max(1.0, (t1 - t0) / 1000.0)


def build_possession_timeline(detections: list[Detection]) -> dict[str, Any]:
    """Vidal-Codina PZ possession on the image plane.

    Critical adaptation: when ball detections are sparse (broadcast highlights
    ~0.4–1 sample/s vs facility ~2–8/s), requiring 3 consecutive assigned
    samples deletes almost every hold. Paper assumes continuous tracking —
    we keep the PZ + contact proof, but lower the temporal run / grace bars
    when density is low so possession changes still form.
    """
    persons = [d for d in detections if d.class_name == "person"]
    balls = [d for d in detections if d.class_name == "sports_ball"]
    ball_series = merge_ball_series(balls)
    if not persons or not ball_series:
        return {"ball_samples": len(ball_series), "segments": [], "holders": []}

    person_index = build_person_index(persons)
    diag = frame_diag(ball_series[0])
    density = ball_density_per_s(ball_series)
    # Sparse ball → slightly larger PZ (broadcast scale) + softer temporal gates
    r_frac = POSSESSION_RADIUS_FRAC if density >= 2.0 else min(0.22, POSSESSION_RADIUS_FRAC + 0.06)
    radius = max(70.0, r_frac * diag)
    # Sparse: min_run=2 keeps short real holds, drops single-frame flybys.
    # Dense facility: keep paper-like run of 3.
    if density >= 2.0:
        min_run = MIN_POSSESSION_SAMPLES
    elif density >= 0.8:
        min_run = 2
    else:
        min_run = 2
    grace_ms = REENTRY_GRACE_MS if density >= 2.0 else max(REENTRY_GRACE_MS, 2500)

    raw = possession_timeline(ball_series, person_index, radius)
    # Contact filter needs 3 local ball samples — skip when too sparse (would null everything)
    if density >= 1.5:
        raw = apply_contact_filter(raw, ball_series, diag)
    smoothed = smooth_holders(raw, min_run=min_run)
    smoothed = apply_reentry_grace(smoothed, grace_ms=grace_ms)
    segments = possession_segments(smoothed)
    return {
        "ball_samples": len(ball_series),
        "diag": diag,
        "radius": radius,
        "ball_density_per_s": round(density, 3),
        "min_run": min_run,
        "segments": [{"holder": h, "start_ms": s, "end_ms": e} for h, s, e in segments],
        "holders": [{"ts_ms": t["ts_ms"], "holder": t.get("holder")} for t in smoothed],
    }


def merge_ball_series(balls: list[Detection]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for b in balls:
        for pt in b.bbox_series or []:
            points.append({**pt, "track_key": b.track_key})
    points.sort(key=lambda p: p.get("ts_ms", 0))
    out: list[dict[str, Any]] = []
    for pt in points:
        if out and out[-1].get("ts_ms") == pt.get("ts_ms"):
            if pt.get("confidence", 0) > out[-1].get("confidence", 0):
                out[-1] = pt
            continue
        out.append(pt)
    return out


def _median_box_frac(series: list[dict[str, Any]]) -> float:
    fracs: list[float] = []
    for pt in series:
        w = float(pt.get("frame_w") or 0) or 1280.0
        h = float(pt.get("frame_h") or 0) or 720.0
        bw = max(0.0, float(pt.get("x2", 0)) - float(pt.get("x1", 0)))
        bh = max(0.0, float(pt.get("y2", 0)) - float(pt.get("y1", 0)))
        fracs.append((bw * bh) / (w * h + 1e-6))
    if not fracs:
        return 0.0
    fracs.sort()
    return fracs[len(fracs) // 2]


def build_person_index(persons: list[Detection]) -> dict[str, list[dict[str, Any]]]:
    """Index person tracks; skip broadcast close-ups (manager / bench faces)."""
    CLOSEUP_BOX_FRAC = 0.18
    index: dict[str, list[dict[str, Any]]] = {}
    for p in persons:
        series = sorted(p.bbox_series or [], key=lambda s: s.get("ts_ms", 0))
        if not series:
            continue
        if _median_box_frac(series) > CLOSEUP_BOX_FRAC:
            continue
        index[p.track_key] = series
    # If everything was a close-up (rare), keep all so we still emit something
    if not index:
        for p in persons:
            series = sorted(p.bbox_series or [], key=lambda s: s.get("ts_ms", 0))
            if series:
                index[p.track_key] = series
    return index


def frame_diag(pt: dict[str, Any]) -> float:
    w = float(pt.get("frame_w") or 1280)
    h = float(pt.get("frame_h") or 720)
    return (w * w + h * h) ** 0.5


def center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def person_at(series: list[dict[str, Any]], ts_ms: int) -> dict[str, Any] | None:
    if not series:
        return None
    best = min(series, key=lambda s: abs(s.get("ts_ms", 0) - ts_ms))
    if abs(best.get("ts_ms", 0) - ts_ms) > PERSON_TIME_WINDOW_MS:
        return None
    return best


def ball_to_player_dist(ball: dict[str, Any], person: dict[str, Any]) -> float:
    bx, by = center(ball)
    x1, y1 = float(person.get("x1", 0)), float(person.get("y1", 0))
    x2, y2 = float(person.get("x2", 0)), float(person.get("y2", 0))
    pad_x = 0.15 * max(1.0, x2 - x1)
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1 = y1 + 0.45 * (y2 - y1)
    y2 = y2 + 0.1 * max(1.0, y2 - y1)
    cx = min(max(bx, x1), x2)
    cy = min(max(by, y1), y2)
    return ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5


def possession_timeline(
    ball_series: list[dict[str, Any]],
    person_index: dict[str, list[dict[str, Any]]],
    radius: float,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for ball in ball_series:
        ts = int(ball.get("ts_ms", 0))
        best_key = None
        best_dist = radius
        for key, series in person_index.items():
            sample = person_at(series, ts)
            if not sample:
                continue
            dist = ball_to_player_dist(ball, sample)
            if dist <= best_dist:
                best_dist = dist
                best_key = key
        timeline.append({"ts_ms": ts, "holder": best_key, "ball": ball, "dist": best_dist})
    return timeline


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        return (0.0, 0.0)
    return (dx / n, dy / n)


def apply_contact_filter(
    timeline: list[dict[str, Any]], ball_series: list[dict[str, Any]], diag: float
) -> list[dict[str, Any]]:
    """Drop proximity-only holds with no Δv/Δdirection (fly-bys)."""
    if len(ball_series) < 3:
        return timeline
    by_ts = {int(p.get("ts_ms", 0)): p for p in ball_series}
    out: list[dict[str, Any]] = []
    i = 0
    n = len(timeline)
    while i < n:
        h = timeline[i].get("holder")
        j = i
        while j < n and timeline[j].get("holder") == h:
            j += 1
        if h is None:
            out.extend(timeline[i:j])
            i = j
            continue
        # Check contact within this control run
        contacted = False
        for k in range(i, j):
            ts = int(timeline[k]["ts_ms"])
            # neighbors in ball series
            idxs = [m for m, p in enumerate(ball_series) if int(p.get("ts_ms", 0)) == ts]
            if not idxs:
                continue
            mi = idxs[0]
            if mi <= 0 or mi >= len(ball_series) - 1:
                continue
            a, b, c = ball_series[mi - 1], ball_series[mi], ball_series[mi + 1]
            ax, ay = center(a)
            bx, by = center(b)
            cx, cy = center(c)
            d0 = _unit(bx - ax, by - ay)
            d1 = _unit(cx - bx, cy - by)
            dot = d0[0] * d1[0] + d0[1] * d1[1]
            dt0 = max(1, int(b.get("ts_ms", 0)) - int(a.get("ts_ms", 0)))
            dt1 = max(1, int(c.get("ts_ms", 0)) - int(b.get("ts_ms", 0)))
            v0 = (((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / diag) / dt0
            v1 = (((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5 / diag) / dt1
            if (1.0 - max(-1.0, min(1.0, dot))) >= MIN_DIR_DOT_CHANGE or abs(v1 - v0) >= MIN_SPEED_CHANGE_FRAC:
                contacted = True
                break
        if contacted or (j - i) >= MIN_POSSESSION_SAMPLES * 2:
            out.extend(timeline[i:j])
        else:
            for k in range(i, j):
                out.append({**timeline[k], "holder": None})
        i = j
    return out


def smooth_holders(timeline: list[dict[str, Any]], min_run: int) -> list[dict[str, Any]]:
    if not timeline:
        return timeline
    holders = [t.get("holder") for t in timeline]
    n = len(holders)
    i = 0
    while i < n:
        h = holders[i]
        j = i
        while j < n and holders[j] == h:
            j += 1
        if h is not None and (j - i) < min_run:
            for k in range(i, j):
                holders[k] = None
        i = j
    return [{**t, "holder": h} for t, h in zip(timeline, holders)]


def apply_reentry_grace(
    timeline: list[dict[str, Any]], grace_ms: int | None = None
) -> list[dict[str, Any]]:
    """If holder A briefly loses then regains within grace with no other holder → keep A."""
    if not timeline:
        return timeline
    limit = REENTRY_GRACE_MS if grace_ms is None else grace_ms
    holders = [t.get("holder") for t in timeline]
    n = len(holders)
    i = 0
    while i < n:
        if holders[i] is None:
            i += 1
            continue
        # find end of A run
        a = holders[i]
        j = i
        while j < n and holders[j] == a:
            j += 1
        # gap of None
        k = j
        while k < n and holders[k] is None:
            k += 1
        if k < n and holders[k] == a and j < n:
            gap_ms = int(timeline[k]["ts_ms"]) - int(timeline[j - 1]["ts_ms"])
            if 0 < gap_ms <= limit:
                for t in range(j, k):
                    holders[t] = a
        i = j if j > i else i + 1
    return [{**t, "holder": h} for t, h in zip(timeline, holders)]


def possession_segments(timeline: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    cur: str | None = None
    start = 0
    for t in timeline:
        h = t.get("holder")
        ts = int(t["ts_ms"])
        if h is None:
            if cur is not None:
                if ts - start >= MIN_POSSESSION_MS:
                    segments.append((cur, start, ts))
                cur = None
            continue
        if h != cur:
            if cur is not None and ts - start >= MIN_POSSESSION_MS:
                segments.append((cur, start, ts))
            cur = h
            start = ts
    if cur is not None and timeline:
        end = int(timeline[-1]["ts_ms"])
        if end - start >= MIN_POSSESSION_MS:
            segments.append((cur, start, end))
    return segments
