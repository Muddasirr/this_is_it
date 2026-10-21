"""Possession-centric event inference (image-plane CCTV).

Aligned with Vidal-Codina et al. 2022 (arXiv:2202.00804):
  1) Assign ball possession per sample (possession zone).
  2) Events from possession changes + ball kinematics — not raw motion bursts.

Pass  = sustained holder A → flight → sustained holder B ≠ A (not a shot).
Shot  = sustained holder releases ball goalward at high speed with no receiver
        gaining possession in the flight window (paper: shot precedes goal/save/
        dead-ball; we approximate without GK tracks).
Goal  = after a shot, ball lingers in the extreme goal band (strict; review gate).

Supplements that flooded v16 stats (sparse_ball_strike, kick_motion, proximity
handoffs) are disabled when ball coverage is usable.
"""

from __future__ import annotations

from typing import Any

from app.services.possession import (
    MIN_POSSESSION_MS,
    build_possession_timeline,
    frame_diag,
    merge_ball_series,
)
from app.services.vision import Detection, EventCandidate

MAX_FLIGHT_MS = 3200
# Sparse ball (highlights / occlusions): allow longer A→B gaps so we don't
# zero-out every pass when the ball drops offline between touches.
MAX_FLIGHT_SPARSE_MS = 25000  # highlights often have 5–20s ball dropouts between touches
MIN_FLIGHT_MS = 0  # adjacent handoff (flight==0) is a completed short pass
# ISVP.AI futsal shot detector (icSPORTS 2023): high acceleration + near player
# + goalward factor + no direction change. Tuned for wide CCTV (frac of diag).
SHOT_SPEED_FRAC_PER_MS = 0.00012
SHOT_MIN_DISP_FRAC = 0.08
SHOT_ACCEL_FRAC_PER_MS2 = 0.00000018
SHOT_KIN_WINDOW_MS = 1800  # evaluate strike over fixed window (not clipped to next touch)
GOAL_BAND_FRAC = 0.10
GOAL_LINGER_SAMPLES = 4
# Quick takeover ⇒ pass/block unless ball already entered goal band (ISVP on-target)
SHOT_MAX_RECEIVER_MS = 850
PASS_DEDUP_MS = 1800
SHOT_DEDUP_MS = 3500
GOAL_DEDUP_MS = 10000
MAX_PASSES = 120
MAX_SHOTS = 18
MAX_GOALS = 8
# Need this many ball samples before trusting possession events alone
MIN_BALL_FOR_STRICT = 80


def infer_event_candidates(
    detections: list[Detection],
    camera_label: str,
) -> list[EventCandidate]:
    persons = [d for d in detections if d.class_name == "person"]
    balls = [d for d in detections if d.class_name == "sports_ball"]
    ball_series = merge_ball_series(balls)
    if not persons:
        return []

    # Without enough ball, do NOT invent shot/pass floods — return empty estimated set
    if len(ball_series) < MIN_BALL_FOR_STRICT:
        return []

    poss = build_possession_timeline(detections)
    segments = [
        (s["holder"], int(s["start_ms"]), int(s["end_ms"])) for s in (poss.get("segments") or [])
    ]
    diag = float(poss.get("diag") or frame_diag(ball_series[0]))

    flight_limit = _pass_flight_limit_ms(ball_series)
    # Shots first, then passes — a release that finds a teammate is a pass, not a shot.
    shots = _shots_from_segments(
        segments, ball_series, camera_label, diag, receiver_window_ms=flight_limit
    )
    passes = _passes_from_segments(segments, ball_series, camera_label, diag, flight_limit)
    shot_ts = {s.camera_ts_ms for s in shots}
    # Drop pass that double-counts the same release already labeled a shot
    passes = [p for p in passes if p.camera_ts_ms not in shot_ts]
    candidates: list[EventCandidate] = []
    candidates.extend(passes)
    candidates.extend(shots)
    candidates.extend(_goals_after_shots(shots, ball_series, camera_label))
    return _dedupe(candidates)


def _pass_flight_limit_ms(ball_series: list[dict[str, Any]]) -> int:
    """Widen pass flight window when ball tracking is sparse (highlights/CCTV gaps)."""
    if len(ball_series) < 2:
        return MAX_FLIGHT_MS
    t0 = min(int(p.get("ts_ms", 0)) for p in ball_series)
    t1 = max(int(p.get("ts_ms", 0)) for p in ball_series)
    span_s = max(1.0, (t1 - t0) / 1000.0)
    density = len(ball_series) / span_s  # samples / second
    # Facility denser ball ≈ 3–8/s; highlights often < 2/s
    if density < 2.0:
        return MAX_FLIGHT_SPARSE_MS
    return MAX_FLIGHT_MS


def _center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def _center_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _ball_speed(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> float:
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 2:
        before = [p for p in ball_series if int(p.get("ts_ms", 0)) <= t0]
        after = [p for p in ball_series if int(p.get("ts_ms", 0)) >= t1]
        if not before or not after:
            return 0.0
        pts = [before[-1], after[0]]
    a, b = pts[0], pts[-1]
    dt = max(1, int(b.get("ts_ms", 0)) - int(a.get("ts_ms", 0)))
    return (_center_delta(a, b) / diag) / dt


def _flight_disp_frac(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> float:
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 2:
        before = [p for p in ball_series if int(p.get("ts_ms", 0)) <= t0]
        after = [p for p in ball_series if int(p.get("ts_ms", 0)) >= t1]
        if not before or not after:
            return 0.0
        pts = [before[-1], after[0]]
    return _center_delta(pts[0], pts[-1]) / max(diag, 1.0)


def _toward_goal(ball_series: list[dict[str, Any]], t0: int, t1: int) -> bool:
    """Facility CCTV goals may sit on any frame edge — check all four bands."""
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 2:
        before = [p for p in ball_series if int(p.get("ts_ms", 0)) <= t0]
        after = [p for p in ball_series if int(p.get("ts_ms", 0)) >= t1]
        if not before or not after:
            return False
        pts = [before[-1], after[0]]
    a, b = pts[0], pts[-1]
    ax, ay = _center(a)
    bx, by = _center(b)
    h = float(b.get("frame_h") or a.get("frame_h") or 720)
    w = float(b.get("frame_w") or a.get("frame_w") or 1280)
    by_band = GOAL_BAND_FRAC * h
    bx_band = GOAL_BAND_FRAC * w
    top = by < by_band and ay > by
    bottom = by > h - by_band and by > ay
    left = bx < bx_band and ax > bx
    right = bx > w - bx_band and bx > ax
    return top or bottom or left or right


def _ball_accel(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> float:
    """Peak |Δspeed| over consecutive samples in [t0,t1] (frac diag / ms²)."""
    pts = [p for p in ball_series if t0 - 200 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 3:
        return 0.0
    best = 0.0
    for i in range(len(pts) - 2):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        dt1 = max(1, int(b.get("ts_ms", 0)) - int(a.get("ts_ms", 0)))
        dt2 = max(1, int(c.get("ts_ms", 0)) - int(b.get("ts_ms", 0)))
        v1 = (_center_delta(a, b) / diag) / dt1
        v2 = (_center_delta(b, c) / diag) / dt2
        best = max(best, abs(v2 - v1) / max(dt1, dt2))
    return best


def _flight_straight(ball_series: list[dict[str, Any]], t0: int, t1: int) -> bool:
    """ISVP: no direction change over early flight samples (≈3 frames)."""
    pts = [p for p in ball_series if t0 <= int(p.get("ts_ms", 0)) <= t1]
    if len(pts) < 4:
        return True
    pts = pts[:6]
    angles: list[float] = []
    for i in range(len(pts) - 1):
        ax, ay = _center(pts[i])
        bx, by = _center(pts[i + 1])
        dx, dy = bx - ax, by - ay
        if abs(dx) + abs(dy) < 1.0:
            continue
        angles.append(_atan2(dy, dx))
    if len(angles) < 2:
        return True
    for i in range(len(angles) - 1):
        d = abs(angles[i + 1] - angles[i])
        if d > 3.14159265:
            d = 6.2831853 - d
        if d > 0.85:  # ~50° turn ⇒ pass/deflection, not clean shot
            return False
    return True


def _atan2(y: float, x: float) -> float:
    import math

    return math.atan2(y, x)


def _is_shot_kinematics(ball_series: list[dict[str, Any]], t0: int, t1: int, diag: float) -> bool:
    """Hard strike: speed + displacement; accel OR very high speed (ISVP-inspired)."""
    speed = _ball_speed(ball_series, t0, t1, diag)
    disp = _flight_disp_frac(ball_series, t0, t1, diag)
    if speed < SHOT_SPEED_FRAC_PER_MS or disp < SHOT_MIN_DISP_FRAC:
        return False
    accel = _ball_accel(ball_series, t0, min(t0 + 600, t1), diag)
    if accel < SHOT_ACCEL_FRAC_PER_MS2 and speed < SHOT_SPEED_FRAC_PER_MS * 1.5:
        return False
    # Prefer straight flights; allow mild turns if clearly on-target speed
    if not _flight_straight(ball_series, t0, t1) and speed < SHOT_SPEED_FRAC_PER_MS * 1.8:
        return False
    return True


def _passes_from_segments(
    segments: list[tuple[str, int, int]],
    ball_series: list[dict[str, Any]],
    camera_label: str,
    diag: float,
    flight_limit_ms: int | None = None,
) -> list[EventCandidate]:
    max_flight = MAX_FLIGHT_MS if flight_limit_ms is None else flight_limit_ms
    candidates: list[EventCandidate] = []
    for i in range(len(segments) - 1):
        a_key, a_start, a_end = segments[i]
        b_key, b_start, b_end = segments[i + 1]
        if a_key == b_key:
            continue
        flight = b_start - a_end
        # flight 0 / small overlap = handoff; large negative is messy ordering — skip
        if flight < -250:
            continue
        if flight > max_flight:
            continue
        # Hard strike veto — only when ball tracking is dense enough for kinematics.
        # On highlights (sparse ball) this falsely kills every handoff.
        if max_flight <= MAX_FLIGHT_MS:
            if _is_shot_kinematics(ball_series, a_end, a_end + SHOT_KIN_WINDOW_MS, diag):
                continue
        # Require both sides held briefly (completed pass)
        if (a_end - a_start) < MIN_POSSESSION_MS:
            continue
        if (b_end - b_start) < MIN_POSSESSION_MS:
            continue
        conf = 0.62
        if flight < 2000:
            conf += 0.08
        if flight > MAX_FLIGHT_MS:
            conf -= 0.08  # sparse-ball / long-gap pass — lower trust
        candidates.append(
            EventCandidate(
                event_type="pass",
                camera_label=camera_label,
                camera_ts_ms=a_end,
                confidence=min(0.88, max(0.45, conf)),
                track_keys=[a_key, b_key],
                evidence={
                    "heuristic": "possession_change",
                    "passer": a_key,
                    "receiver": b_key,
                    "flight_ms": flight,
                    "possession_a_ms": a_end - a_start,
                    "possession_b_ms": b_end - b_start,
                    "flight_limit_ms": max_flight,
                    "method": "vidal_codina_pass",
                },
            )
        )
        if len(candidates) >= MAX_PASSES:
            break
    return candidates


def _shots_from_segments(
    segments: list[tuple[str, int, int]],
    ball_series: list[dict[str, Any]],
    camera_label: str,
    diag: float,
    receiver_window_ms: int | None = None,
) -> list[EventCandidate]:
    """Shot ≈ ISVP futsal detector + Vidal-Codina possession release.

    Count when: hard goalward release AND (no teammate gains possession in the
    pass-flight window OR ball reaches goal band — on-target / save / goal).
    A completed pass to a teammate is never also a shot.
    """
    rx_window = SHOT_MAX_RECEIVER_MS if receiver_window_ms is None else max(
        SHOT_MAX_RECEIVER_MS, receiver_window_ms
    )
    candidates: list[EventCandidate] = []
    for i, (holder, start, end) in enumerate(segments):
        if (end - start) < MIN_POSSESSION_MS:
            continue
        if i + 1 < len(segments):
            next_holder, next_start, _ = segments[i + 1]
            teammate_rx = next_holder != holder and 0 <= (next_start - end) <= rx_window
        else:
            teammate_rx = False

        kin_end = end + SHOT_KIN_WINDOW_MS
        if not _is_shot_kinematics(ball_series, end, kin_end, diag):
            continue
        if not _toward_goal(ball_series, end, kin_end):
            continue

        on_target = _ball_enters_goal_band(ball_series, end)
        if teammate_rx:
            # Another on-court player gained the ball → pass / block / rebound
            # chain, not a shot. (Edge "goal band" fires too often on TV cuts.)
            continue


        speed = _ball_speed(ball_series, end, kin_end, diag)
        candidates.append(
            EventCandidate(
                event_type="shot",
                camera_label=camera_label,
                camera_ts_ms=end,
                confidence=0.72 if on_target else 0.60,
                track_keys=[holder],
                evidence={
                    "heuristic": "possession_release_shot",
                    "shooter": holder,
                    "goalward": True,
                    "on_target_band": on_target,
                    "speed_diag_per_ms": speed,
                    "method": "vidal_codina_isvp_shot",
                    "possession_ms": end - start,
                },
            )
        )
        if len(candidates) >= MAX_SHOTS:
            break
    return candidates


def _ball_enters_goal_band(
    ball_series: list[dict[str, Any]], t0: int, window_ms: int = 2200
) -> bool:
    """True if ball reaches a deep edge band soon after release (on-target)."""
    deep = GOAL_BAND_FRAC * 0.75
    for p in ball_series:
        ts = int(p.get("ts_ms", 0))
        if ts < t0 or ts > t0 + window_ms:
            continue
        cx, cy = _center(p)
        h = float(p.get("frame_h") or 720)
        w = float(p.get("frame_w") or 1280)
        if (
            cy < deep * h
            or cy > (1 - deep) * h
            or cx < deep * w
            or cx > (1 - deep) * w
        ):
            return True
    return False


def _ball_lingers_in_goal(
    ball_series: list[dict[str, Any]], t0: int, window_ms: int = 2500
) -> bool:
    band_hits = 0
    for p in ball_series:
        ts = int(p.get("ts_ms", 0))
        if ts < t0 or ts > t0 + window_ms:
            continue
        cx, cy = _center(p)
        h = float(p.get("frame_h") or 720)
        w = float(p.get("frame_w") or 1280)
        in_band = (
            cy < GOAL_BAND_FRAC * h
            or cy > (1 - GOAL_BAND_FRAC) * h
            or cx < GOAL_BAND_FRAC * w
            or cx > (1 - GOAL_BAND_FRAC) * w
        )
        if in_band:
            band_hits += 1
            if band_hits >= GOAL_LINGER_SAMPLES:
                return True
    return False


def _goals_after_shots(
    shots: list[EventCandidate],
    ball_series: list[dict[str, Any]],
    camera_label: str,
) -> list[EventCandidate]:
    """Goal only after a goalward shot with ball lingering in goal band (strict)."""
    candidates: list[EventCandidate] = []
    for shot in shots:
        if not (shot.evidence or {}).get("goalward"):
            continue
        t0 = shot.camera_ts_ms
        if not _ball_lingers_in_goal(ball_series, t0):
            continue
        # Extra gate: must reach deep band (reduces sideline false goals)
        if not _ball_enters_goal_band(ball_series, t0, window_ms=2000):
            continue
        candidates.append(
            EventCandidate(
                event_type="goal",
                camera_label=camera_label,
                camera_ts_ms=t0 + 400,
                confidence=0.40,
                track_keys=list(shot.track_keys),
                evidence={
                    "heuristic": "shot_then_goal_band_linger",
                    "shooter": shot.track_keys[0] if shot.track_keys else None,
                    "method": "vidal_codina_goal_approx",
                    "note": "estimated — confirm in review",
                },
            )
        )
        if len(candidates) >= MAX_GOALS:
            break
    return candidates


def _dedupe(candidates: list[EventCandidate]) -> list[EventCandidate]:
    seen: set[tuple] = set()
    out: list[EventCandidate] = []
    for c in sorted(candidates, key=lambda x: (-x.confidence, x.camera_ts_ms)):
        if c.event_type == "pass":
            bucket = c.camera_ts_ms // PASS_DEDUP_MS
        elif c.event_type == "shot":
            bucket = c.camera_ts_ms // SHOT_DEDUP_MS
        elif c.event_type == "goal":
            bucket = c.camera_ts_ms // GOAL_DEDUP_MS
        else:
            bucket = c.camera_ts_ms // PASS_DEDUP_MS
        key = (c.event_type, bucket, tuple(sorted(c.track_keys)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
