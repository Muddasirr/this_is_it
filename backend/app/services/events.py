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

from contextvars import ContextVar
from typing import Any

from app.services.possession import (
    MIN_POSSESSION_MS,
    ball_density_per_s,
    build_person_index,
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
# Flood guards, scaled with video length: a fixed MAX_PASSES = 120 cut a 12-min
# SoccerTrack half (167 labelled passes) at ~115 and would drop ~80 % of a full half.
MAX_PASSES = 120
MAX_SHOTS = 18
MAX_GOALS = 8
MAX_PER_MIN = {"pass": 40.0, "shot": 3.0, "goal": 1.0, "dribble": 8.0, "assist": 1.0}
_LIMITS: ContextVar[dict[str, int] | None] = ContextVar("event_limits", default=None)


def _limit(kind: str, default: int) -> int:
    limits = _LIMITS.get()
    return max(default, limits.get(kind, default)) if limits else default
# Need this many ball samples before trusting possession events alone
MIN_BALL_FOR_STRICT = 80
# Short clips / short spells: enough ball if it is dense, even below 80 samples
# (a 2.5 s labelled clip has ≤ 63 frames — the absolute gate rejected every one).
MIN_BALL_DENSE_SAMPLES = 10
# A pass moves the ball: holder flips with the ball standing still are challenges,
# shielding or tracker ID switches (tuned on DFL labelled clips — see bench_events).
MIN_PASS_BALL_DISP_FRAC = 0.0
# Passer's last box ≈ receiver's first box ⇒ the tracker re-IDed one player, not a pass.
ID_SWAP_IOU = 1.01
MIN_BALL_DENSE_PER_S = 2.0


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
    if not _enough_ball(ball_series):
        return []
    minutes = max(0.0, (int(ball_series[-1].get("ts_ms", 0)) - int(ball_series[0].get("ts_ms", 0))) / 60000.0)
    token = _LIMITS.set({k: int(v * minutes) for k, v in MAX_PER_MIN.items()})
    try:
        return _infer(detections, persons, ball_series, camera_label)
    finally:
        _LIMITS.reset(token)


def _infer(detections, persons, ball_series, camera_label: str) -> list[EventCandidate]:
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
    persons_index = build_person_index(persons)
    passes = _passes_from_segments(segments, ball_series, camera_label, diag, flight_limit, persons_index)
    shot_ts = {s.camera_ts_ms for s in shots}
    # Drop pass that double-counts the same release already labeled a shot
    passes = [p for p in passes if p.camera_ts_ms not in shot_ts]
    goals = _goals_after_shots(shots, ball_series, camera_label)
    dribbles = _dribbles_from_segments(segments, persons, camera_label, diag)
    assists = _assists_from_goals(goals, passes, camera_label)
    candidates: list[EventCandidate] = []
    candidates.extend(passes)
    candidates.extend(shots)
    candidates.extend(goals)
    candidates.extend(assists)
    candidates.extend(dribbles)
    return _dedupe(candidates)


ASSIST_WINDOW_MS = 20000  # facility CCTV: longer chain than broadcast tracking
DRIBBLE_MIN_MS = 1200
DRIBBLE_MIN_DISP_FRAC = 0.06
MAX_DRIBBLES = 40
MAX_ASSISTS = 12


def _assists_from_goals(
    goals: list[EventCandidate],
    passes: list[EventCandidate],
    camera_label: str,
) -> list[EventCandidate]:
    """Pass whose receiver scores within ASSIST_WINDOW_MS → assist (passer).

    Vidal-Codina style: events sit on possession chain; assist is the prior
    completed pass into the scorer (estimated until review).
    """
    out: list[EventCandidate] = []
    for g in goals:
        gt = g.camera_ts_ms
        scorer = (g.track_keys or [None])[0]
        best = None
        best_dt = ASSIST_WINDOW_MS + 1
        for p in passes:
            pt = p.camera_ts_ms
            if pt >= gt or gt - pt > ASSIST_WINDOW_MS:
                continue
            keys = p.track_keys or []
            # pass evidence often has receiver; else second track key
            receiver = (p.evidence or {}).get("receiver") or (keys[1] if len(keys) > 1 else None)
            passer = (p.evidence or {}).get("passer") or (keys[0] if keys else None)
            if scorer and receiver and receiver != scorer:
                continue
            if not passer or passer == scorer:
                continue
            dt = gt - pt
            if dt < best_dt:
                best_dt = dt
                best = (passer, p)
        if best is None:
            continue
        passer, src_pass = best
        out.append(
            EventCandidate(
                event_type="assist",
                camera_label=camera_label,
                camera_ts_ms=src_pass.camera_ts_ms,
                confidence=min(0.55, 0.35 + src_pass.confidence * 0.3),
                track_keys=[passer],
                evidence={
                    "heuristic": "pass_then_goal",
                    "passer": passer,
                    "scorer": scorer,
                    "goal_ts_ms": gt,
                    "method": "possession_chain_assist",
                },
            )
        )
        if len(out) >= _limit("assist", MAX_ASSISTS):
            break
    return out


def _dribbles_from_segments(
    segments: list[tuple[str, int, int]],
    persons: list[Detection],
    camera_label: str,
    diag: float,
) -> list[EventCandidate]:
    """Progressive carry: same holder keeps ball while moving (≥DRIBBLE_MIN_DISP).

    Not in Vidal-Codina core taxonomy; useful futsal/facility metric (estimated).
    """
    by_key = {d.track_key: d for d in persons}
    out: list[EventCandidate] = []
    for holder, t0, t1 in segments:
        if t1 - t0 < DRIBBLE_MIN_MS:
            continue
        det = by_key.get(holder)
        if not det:
            continue
        series = sorted(det.bbox_series or [], key=lambda p: int(p.get("ts_ms") or 0))
        pts = [p for p in series if t0 <= int(p.get("ts_ms") or 0) <= t1]
        if len(pts) < 3:
            continue
        disp = _center_delta(pts[0], pts[-1]) / max(diag, 1.0)
        if disp < DRIBBLE_MIN_DISP_FRAC:
            continue
        out.append(
            EventCandidate(
                event_type="dribble",
                camera_label=camera_label,
                camera_ts_ms=t0 + (t1 - t0) // 2,
                confidence=0.45,
                track_keys=[holder],
                evidence={
                    "heuristic": "possession_progressive_carry",
                    "holder": holder,
                    "duration_ms": t1 - t0,
                    "disp_frac": round(disp, 4),
                    "method": "facility_dribble_approx",
                },
            )
        )
        if len(out) >= _limit("dribble", MAX_DRIBBLES):
            break
    return out


def _enough_ball(ball_series: list[dict]) -> bool:
    if len(ball_series) >= MIN_BALL_FOR_STRICT:
        return True
    return len(ball_series) >= MIN_BALL_DENSE_SAMPLES and ball_density_per_s(ball_series) >= MIN_BALL_DENSE_PER_S


def _ball_at(ball_series: list[dict[str, Any]], ts_ms: int, window_ms: int = 400) -> tuple[float, float] | None:
    best = min(ball_series, key=lambda p: abs(int(p.get("ts_ms", 0)) - ts_ms), default=None)
    if best is None or abs(int(best.get("ts_ms", 0)) - ts_ms) > window_ms:
        return None
    return ((best["x1"] + best["x2"]) / 2.0, (best["y1"] + best["y2"]) / 2.0)


def _ball_displacement(ball_series: list[dict[str, Any]], t_release: int, t_receive: int) -> float | None:
    a, b = _ball_at(ball_series, t_release), _ball_at(ball_series, t_receive)
    if a is None or b is None:
        return None
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _same_player_reid(persons_index, a_key: str, b_key: str, t_a: int, t_b: int) -> bool:
    """Passer track ends where the receiver track starts, never co-existing ⇒ one player re-IDed."""
    if ID_SWAP_IOU > 1.0:
        return False
    from app.services.possession import person_at

    sa, sb = persons_index.get(a_key) or [], persons_index.get(b_key) or []
    if not sa or not sb:
        return False
    overlap = min(sa[-1]["ts_ms"], sb[-1]["ts_ms"]) - max(sa[0]["ts_ms"], sb[0]["ts_ms"])
    if overlap > 200:  # both tracks on screen together ⇒ two people
        return False
    pa, pb = person_at(sa, t_a), person_at(sb, t_b)
    if pa is None or pb is None:
        return False
    ix = max(0.0, min(pa["x2"], pb["x2"]) - max(pa["x1"], pb["x1"]))
    iy = max(0.0, min(pa["y2"], pb["y2"]) - max(pa["y1"], pb["y1"]))
    inter = ix * iy
    union = (pa["x2"] - pa["x1"]) * (pa["y2"] - pa["y1"]) + (pb["x2"] - pb["x1"]) * (pb["y2"] - pb["y1"]) - inter
    return union > 0 and inter / union >= ID_SWAP_IOU


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
    persons_index: dict[str, list[dict[str, Any]]] | None = None,
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
        # Receiver already gained possession → completed pass (Vidal-Codina).
        # Do NOT veto on strike kinematics: lateral/through balls are fast but
        # still passes when B holds. Shots are handled separately when no receiver.
        # Require both sides held briefly (completed pass)
        if (a_end - a_start) < MIN_POSSESSION_MS:
            continue
        if (b_end - b_start) < MIN_POSSESSION_MS:
            continue
        disp = _ball_displacement(ball_series, a_end, b_start)
        # Reject only when the ball is seen standing still; unknown (ball missed) is not evidence
        if MIN_PASS_BALL_DISP_FRAC > 0 and disp is not None and disp < MIN_PASS_BALL_DISP_FRAC * diag:
            continue
        if persons_index and _same_player_reid(persons_index, a_key, b_key, a_end, b_start):
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
        if len(candidates) >= _limit("pass", MAX_PASSES):
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
        if len(candidates) >= _limit("shot", MAX_SHOTS):
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
        if len(candidates) >= _limit("goal", MAX_GOALS):
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
        elif c.event_type == "assist":
            bucket = c.camera_ts_ms // ASSIST_WINDOW_MS
        elif c.event_type == "dribble":
            bucket = c.camera_ts_ms // DRIBBLE_MIN_MS
        else:
            bucket = c.camera_ts_ms // PASS_DEDUP_MS
        key = (c.event_type, bucket, tuple(sorted(c.track_keys)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
