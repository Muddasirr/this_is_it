"""Possession-centric event inference (league-style, image-plane).

Adapted from optical-tracking event literature (possession zones → gains/losses
→ classify as pass/shot/goal). Operates on YOLO/ByteTrack image coordinates —
not calibrated pitch meters. See ARCHITECTURE.md.
"""

from __future__ import annotations

from typing import Any

from app.services.vision import Detection, EventCandidate

# Relative to frame diagonal — generous for wide CCTV / YOLO foot-ball offset
POSSESSION_RADIUS_FRAC = 0.18
MIN_POSSESSION_SAMPLES = 2
MAX_FLIGHT_MS = 5000
MIN_FLIGHT_MS = 80
SHOT_SPEED_FRAC_PER_MS = 0.00028  # diagonal fractions per ms
GOAL_BAND_FRAC = 0.12
PASS_DEDUP_MS = 1200
MAX_PASSES = 400
MAX_SHOTS = 120
PERSON_TIME_WINDOW_MS = 2500


def infer_event_candidates(
    detections: list[Detection],
    camera_label: str,
) -> list[EventCandidate]:
    persons = [d for d in detections if d.class_name == "person"]
    balls = [d for d in detections if d.class_name == "sports_ball"]

    ball_series = _merge_ball_series(balls)
    if not persons:
        return []

    if len(ball_series) < 3:
        return _fallback_without_ball(persons, camera_label)

    person_index = _build_person_index(persons)
    diag = _frame_diag(ball_series[0])
    radius = max(80.0, POSSESSION_RADIUS_FRAC * diag)

    holders = _possession_timeline(ball_series, person_index, radius)
    holders = _smooth_holders(holders, min_run=MIN_POSSESSION_SAMPLES)

    candidates: list[EventCandidate] = []
    candidates.extend(_passes_from_holders(holders, ball_series, camera_label, diag))
    candidates.extend(_shots_and_goals(holders, ball_series, camera_label, diag))
    candidates.extend(_sparse_ball_shots(ball_series, person_index, camera_label, diag))

    # Sparse ball tracks (common on facility CCTV): supplement with proximity handoffs
    passes = [c for c in candidates if c.event_type == "pass"]
    if len(passes) < 15:
        candidates.extend(_fallback_without_ball(persons, camera_label))
    return _dedupe(candidates)


def _merge_ball_series(balls: list[Detection]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for b in balls:
        for pt in b.bbox_series or []:
            points.append({**pt, "track_key": b.track_key})
    points.sort(key=lambda p: p.get("ts_ms", 0))
    # collapse same timestamp
    out: list[dict[str, Any]] = []
    for pt in points:
        if out and out[-1].get("ts_ms") == pt.get("ts_ms"):
            if pt.get("confidence", 0) > out[-1].get("confidence", 0):
                out[-1] = pt
            continue
        out.append(pt)
    return out


def _build_person_index(persons: list[Detection]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for p in persons:
        series = sorted(p.bbox_series or [], key=lambda s: s.get("ts_ms", 0))
        if series:
            index[p.track_key] = series
    return index


def _frame_diag(pt: dict[str, Any]) -> float:
    w = float(pt.get("frame_w") or 1280)
    h = float(pt.get("frame_h") or 720)
    return (w * w + h * h) ** 0.5


def _center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def _person_at(series: list[dict[str, Any]], ts_ms: int) -> dict[str, Any] | None:
    if not series:
        return None
    best = min(series, key=lambda s: abs(s.get("ts_ms", 0) - ts_ms))
    if abs(best.get("ts_ms", 0) - ts_ms) > PERSON_TIME_WINDOW_MS:
        return None
    return best


def _ball_to_player_dist(ball: dict[str, Any], person: dict[str, Any]) -> float:
    """Distance from ball center to player foot/body (expanded bbox), not torso center.

    League systems use calibrated XY; we approximate contact with the lower body.
    """
    bx, by = _center(ball)
    x1, y1 = float(person.get("x1", 0)), float(person.get("y1", 0))
    x2, y2 = float(person.get("x2", 0)), float(person.get("y2", 0))
    # expand box slightly; prefer bottom-third (feet / ball contact)
    pad_x = 0.15 * max(1.0, x2 - x1)
    pad_y = 0.1 * max(1.0, y2 - y1)
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1 = y1 + 0.45 * (y2 - y1)  # ignore head/shoulders
    y2 = y2 + pad_y
    cx = min(max(bx, x1), x2)
    cy = min(max(by, y1), y2)
    return ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5


def _possession_timeline(
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
            sample = _person_at(series, ts)
            if not sample:
                continue
            dist = _ball_to_player_dist(ball, sample)
            if dist <= best_dist:
                best_dist = dist
                best_key = key
        timeline.append(
            {
                "ts_ms": ts,
                "holder": best_key,
                "ball": ball,
                "dist": best_dist if best_key else None,
            }
        )
    return timeline


def _smooth_holders(timeline: list[dict[str, Any]], min_run: int) -> list[dict[str, Any]]:
    """Require min_run consecutive same-holder samples; else treat as None (flight)."""
    if not timeline:
        return timeline
    # forward-fill short gaps of same neighbors
    holders = [t.get("holder") for t in timeline]
    n = len(holders)
    i = 0
    while i < n:
        h = holders[i]
        j = i
        while j < n and holders[j] == h:
            j += 1
        run = j - i
        if h is not None and run < min_run:
            for k in range(i, j):
                holders[k] = None
        i = j
    out = []
    for t, h in zip(timeline, holders):
        out.append({**t, "holder": h})
    return out


def _passes_from_holders(
    timeline: list[dict[str, Any]],
    ball_series: list[dict[str, Any]],
    camera_label: str,
    diag: float,
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    # compress into possession segments
    segments: list[tuple[str, int, int]] = []  # holder, start_ms, end_ms
    cur: str | None = None
    start = 0
    for t in timeline:
        h = t.get("holder")
        ts = int(t["ts_ms"])
        if h is None:
            if cur is not None:
                segments.append((cur, start, ts))
                cur = None
            continue
        if h != cur:
            if cur is not None:
                segments.append((cur, start, ts))
            cur = h
            start = ts
    if cur is not None and timeline:
        segments.append((cur, start, int(timeline[-1]["ts_ms"])))

    for i in range(len(segments) - 1):
        a_key, a_start, a_end = segments[i]
        b_key, b_start, b_end = segments[i + 1]
        if a_key == b_key:
            continue
        flight = b_start - a_end
        if flight < MIN_FLIGHT_MS or flight > MAX_FLIGHT_MS:
            continue
        # skip if this release looks like a shot toward goal
        if _is_shot_release(ball_series, a_end, b_start, diag):
            continue
        conf = 0.55
        if flight < 2500:
            conf += 0.1
        if (b_end - b_start) >= 200:
            conf += 0.05
        candidates.append(
            EventCandidate(
                event_type="pass",
                camera_label=camera_label,
                camera_ts_ms=a_end,
                confidence=min(0.85, conf),
                track_keys=[a_key, b_key],
                evidence={
                    "heuristic": "possession_change",
                    "passer": a_key,
                    "receiver": b_key,
                    "flight_ms": flight,
                    "possession_a_ms": a_end - a_start,
                    "possession_b_ms": b_end - b_start,
                },
            )
        )
        if len(candidates) >= MAX_PASSES:
            break
    return candidates


def _ball_speed(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> float:
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 2:
        # nearest neighbors
        before = [p for p in ball_series if int(p.get("ts_ms", 0)) <= t0]
        after = [p for p in ball_series if int(p.get("ts_ms", 0)) >= t1]
        if not before or not after:
            return 0.0
        pts = [before[-1], after[0]]
    a, b = pts[0], pts[-1]
    dist = _center_delta(a, b)
    dt = max(1, int(b.get("ts_ms", 0)) - int(a.get("ts_ms", 0)))
    return (dist / diag) / dt


def _center_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _toward_goal(ball_series: list[dict[str, Any]], t0: int, t1: int) -> bool:
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 2:
        return False
    a, b = pts[0], pts[-1]
    _, ay = _center(a)
    _, by = _center(b)
    h = float(b.get("frame_h") or a.get("frame_h") or 720)
    # moving into top or bottom goal band
    band = GOAL_BAND_FRAC * h
    entering_top = by < band and ay > by
    entering_bottom = by > h - band and by > ay
    return entering_top or entering_bottom


def _is_shot_release(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> bool:
    speed = _ball_speed(ball_series, t0, t1, diag)
    return speed >= SHOT_SPEED_FRAC_PER_MS and _toward_goal(ball_series, t0, t1)


def _shots_and_goals(
    timeline: list[dict[str, Any]],
    ball_series: list[dict[str, Any]],
    camera_label: str,
    diag: float,
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    # look at holder → None → (optional holder) transitions with fast goalward ball
    i = 0
    while i < len(timeline) - 1:
        t = timeline[i]
        if t.get("holder") is None:
            i += 1
            continue
        # find end of this possession
        j = i
        while j < len(timeline) and timeline[j].get("holder") == t["holder"]:
            j += 1
        release_ts = int(timeline[j - 1]["ts_ms"])
        # flight window
        k = j
        while k < len(timeline) and timeline[k].get("holder") is None:
            k += 1
        if k == j:
            i = j
            continue
        flight_end = int(timeline[min(k, len(timeline) - 1)]["ts_ms"])
        if not _is_shot_release(ball_series, release_ts, flight_end, diag):
            i = j
            continue
        shooter = t["holder"]
        candidates.append(
            EventCandidate(
                event_type="shot",
                camera_label=camera_label,
                camera_ts_ms=release_ts,
                confidence=0.5,
                track_keys=[shooter],
                evidence={
                    "heuristic": "goalward_release",
                    "shooter": shooter,
                    "speed_diag_per_ms": _ball_speed(ball_series, release_ts, flight_end, diag),
                },
            )
        )
        # goal if ball ends in extreme band
        end_pt = min(ball_series, key=lambda p: abs(int(p.get("ts_ms", 0)) - flight_end))
        cy = _center(end_pt)[1]
        h = float(end_pt.get("frame_h") or 720)
        if cy < GOAL_BAND_FRAC * h or cy > (1 - GOAL_BAND_FRAC) * h:
            candidates.append(
                EventCandidate(
                    event_type="goal",
                    camera_label=camera_label,
                    camera_ts_ms=flight_end,
                    confidence=0.4,
                    track_keys=[shooter],
                    evidence={"heuristic": "ball_in_goal_band", "shooter": shooter, "point": end_pt},
                )
            )
        if len([c for c in candidates if c.event_type == "shot"]) >= MAX_SHOTS:
            break
        i = k if k > j else j
    return candidates


def _sparse_ball_shots(
    ball_series: list[dict[str, Any]],
    person_index: dict[str, list[dict[str, Any]]],
    camera_label: str,
    diag: float,
) -> list[EventCandidate]:
    """Detect shots from ball kinematics alone when possession chains are thin."""
    candidates: list[EventCandidate] = []
    if len(ball_series) < 2:
        return candidates
    for i in range(len(ball_series) - 1):
        a, b = ball_series[i], ball_series[i + 1]
        t0, t1 = int(a.get("ts_ms", 0)), int(b.get("ts_ms", 0))
        dt = max(1, t1 - t0)
        if dt > 2500:
            continue
        speed = (_center_delta(a, b) / diag) / dt
        if speed < SHOT_SPEED_FRAC_PER_MS:
            continue
        if not _toward_goal([a, b], t0, t1):
            continue
        # attribute to nearest player at release
        shooter = None
        best = 1e9
        for key, series in person_index.items():
            sample = _person_at(series, t0)
            if not sample:
                continue
            dist = _ball_to_player_dist(a, sample)
            if dist < best:
                best = dist
                shooter = key
        if not shooter or best > 0.25 * diag:
            continue
        candidates.append(
            EventCandidate(
                event_type="shot",
                camera_label=camera_label,
                camera_ts_ms=t0,
                confidence=0.42,
                track_keys=[shooter],
                evidence={
                    "heuristic": "sparse_ball_goalward",
                    "shooter": shooter,
                    "speed_diag_per_ms": speed,
                },
            )
        )
        # goal band landing
        cy = _center(b)[1]
        h = float(b.get("frame_h") or 720)
        if cy < GOAL_BAND_FRAC * h or cy > (1 - GOAL_BAND_FRAC) * h:
            candidates.append(
                EventCandidate(
                    event_type="goal",
                    camera_label=camera_label,
                    camera_ts_ms=t1,
                    confidence=0.35,
                    track_keys=[shooter],
                    evidence={"heuristic": "sparse_ball_goal_band", "shooter": shooter},
                )
            )
        if len([c for c in candidates if c.event_type == "shot"]) >= MAX_SHOTS:
            break
    return candidates


def _fallback_without_ball(persons: list[Detection], camera_label: str) -> list[EventCandidate]:
    """When ball is barely detected: proximity handoffs between person tracks over time."""
    candidates: list[EventCandidate] = []
    # sample centers over time per track
    samples: list[tuple[int, str, float, float]] = []
    for p in persons:
        for pt in p.bbox_series or []:
            cx, cy = _center(pt)
            samples.append((int(pt.get("ts_ms", 0)), p.track_key, cx, cy))
    samples.sort()
    # for each time bucket, find two closest players — weak pass signal when they swap closeness to a midpoint
    # Simpler: when two tracks come within radius then separate, count pass once
    for i, a in enumerate(persons):
        for b in persons[i + 1 :]:
            sa = a.bbox_series or []
            sb = b.bbox_series or []
            if len(sa) < 2 or len(sb) < 2:
                continue
            close_ts = None
            for pa in sa:
                pb = _person_at(sb, int(pa.get("ts_ms", 0)))
                if not pb:
                    continue
                if _center_delta(pa, pb) < 120:
                    close_ts = int(pa.get("ts_ms", 0))
                    break
            if close_ts is None:
                continue
            # later separation
            for pa in sa:
                ts = int(pa.get("ts_ms", 0))
                if ts < close_ts + 400:
                    continue
                pb = _person_at(sb, ts)
                if pb and _center_delta(pa, pb) > 200:
                    candidates.append(
                        EventCandidate(
                            event_type="pass",
                            camera_label=camera_label,
                            camera_ts_ms=close_ts,
                            confidence=0.28,
                            track_keys=[a.track_key, b.track_key],
                            evidence={"heuristic": "proximity_handoff_no_ball"},
                        )
                    )
                    break
            if len(candidates) >= 80:
                return _dedupe(candidates)
    return _dedupe(candidates)


def _dedupe(candidates: list[EventCandidate]) -> list[EventCandidate]:
    seen: set[tuple] = set()
    out: list[EventCandidate] = []
    for c in sorted(candidates, key=lambda x: (-x.confidence, x.camera_ts_ms)):
        bucket = c.camera_ts_ms // PASS_DEDUP_MS
        key = (c.event_type, bucket, tuple(sorted(c.track_keys)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
