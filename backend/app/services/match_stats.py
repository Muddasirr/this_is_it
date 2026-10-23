"""Derive richer match / player stats from tracks + canonical events.

Possession and heatmaps are estimated from image-plane tracking (not calibrated
pitch meters). Always treat as estimated unless staff-confirmed events say otherwise.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import EventParticipant, MatchEvent, Player, PlayerTrack, TrackIdentityLink
from app.models.entities import ConfirmationState, EventType
from app.services.court_calib import mapper, track_motion
from app.services.possession import TrackIndex

HEAT_W = 32
HEAT_H = 18
POSSESSION_RADIUS_FRAC = 0.18
PERSON_TIME_WINDOW_MS = 2500


def build_match_stats(db: Session, match_id: UUID) -> dict[str, Any]:
    events = (
        db.query(MatchEvent)
        .filter(MatchEvent.match_id == match_id, MatchEvent.is_canonical.is_(True))
        .all()
    )
    tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == match_id).all()

    totals = {
        "goal": 0,
        "shot": 0,
        "pass": 0,
        "assist": 0,
        "dribble": 0,
        "confirmed": 0,
        "estimated": 0,
        "rejected": 0,
    }
    by_player: dict[str, dict[str, Any]] = {}

    def row(pid: str) -> dict[str, Any]:
        return by_player.setdefault(
            pid,
            {
                "player_id": pid,
                "goal": 0,
                "shot": 0,
                "pass": 0,
                "assist": 0,
                "dribble": 0,
                "pass_received": 0,
                "possession_pct": 0.0,
                "touches": 0,
                "distance": 0.0,
                "minutes": 0.0,
                "avg_speed": 0.0,
                "distance_m": None,
                "avg_speed_kmh": None,
                "top_speed_kmh": None,
                "conversion_pct": None,
                "heatmap": _empty_heat(),
                "possession_heatmap": _empty_heat(),
                "attribution": "estimated",
            },
        )

    for e in events:
        et = _event_type_value(e.event_type)
        if et in totals and e.confirmation != ConfirmationState.rejected:
            totals[et] += 1
        totals[e.confirmation.value] = totals.get(e.confirmation.value, 0) + 1
        parts = db.query(EventParticipant).filter(EventParticipant.event_id == e.id).all()
        for p in parts:
            if not p.player_id or e.confirmation == ConfirmationState.rejected:
                continue
            if et == "pass" and p.role == "receiver":
                row(str(p.player_id))["pass_received"] += 1
                continue
            if et == "pass" and p.role not in {"passer", "primary"}:
                continue
            if et == "shot" and p.role not in {"shooter", "primary"}:
                continue
            if et == "goal" and p.role not in {"scorer", "primary"}:
                continue
            if et == "assist" and p.role not in {"passer", "assist", "primary"}:
                continue
            if et == "dribble" and p.role not in {"carrier", "dribbler", "primary"}:
                continue
            r = row(str(p.player_id))
            if et in r:
                r[et] += 1
            r["attribution"] = (
                "confirmed" if e.confirmation == ConfirmationState.confirmed else "estimated"
            )

    # Track-derived metrics: distance, heatmap, possession, touches
    track_player = _latest_track_players(db, tracks)
    persons = [t for t in tracks if (t.features or {}).get("class", "person") == "person"]
    balls = [t for t in tracks if (t.features or {}).get("class") == "sports_ball"]

    match_heat = _empty_heat()
    match_poss_heat = _empty_heat()
    calibrations = _camera_calibrations(db, match_id)
    # Per (player, camera): a player seen on 2+ angles must not be summed across them.
    per_cam: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"distance": 0.0, "presence_ms": 0.0, "distance_m": 0.0, "top_speed_kmh": 0.0}
    )
    for t in persons:
        pid = track_player.get(t.id)
        series = (t.features or {}).get("bbox_series") or []
        heat = _series_to_heat(series)
        _add_heat(match_heat, heat)
        if pid:
            cam = getattr(t.camera_label, "value", t.camera_label) or "A"
            agg = per_cam[(pid, cam)]
            agg["distance"] += _path_length(series)
            agg["presence_ms"] += max(0, t.end_ms - t.start_ms)
            if cam in calibrations:
                cal, H = calibrations[cam]
                motion = track_motion(series, cal, H)
                agg["distance_m"] += motion["distance_m"]
                agg["top_speed_kmh"] = max(agg["top_speed_kmh"], motion["top_speed_kmh"])
            _add_heat(row(pid)["heatmap"], heat)
    best_cam: dict[str, tuple[str, dict[str, float]]] = {}
    for (pid, cam), agg in per_cam.items():
        if pid not in best_cam or agg["presence_ms"] > best_cam[pid][1]["presence_ms"]:
            best_cam[pid] = (cam, agg)
    for pid, (cam, agg) in best_cam.items():
        r = row(pid)
        r["distance"] = agg["distance"]
        if cam in calibrations:
            r["distance_m"] = round(agg["distance_m"], 1)
            r["top_speed_kmh"] = agg["top_speed_kmh"]
            minutes = agg["presence_ms"] / 60000.0
            r["avg_speed_kmh"] = round(agg["distance_m"] / 1000.0 / (minutes / 60.0), 1) if minutes > 0 else None

    poss_ms, touches = _possession_and_touches(persons, balls, track_player)
    # Vidal-Codina §3.2.1: possession-informed heatmap (on-ball locations only)
    _accumulate_possession_heat(
        persons, balls, track_player, by_player, row, match_poss_heat
    )
    event_touches = _event_touch_counts(db, events)
    # Blend ball-control with on-camera presence so sparse ball tracking
    # does not give one player ~100% possession.
    presence_ms: dict[str, int] = {pid: int(agg["presence_ms"]) for pid, (_cam, agg) in best_cam.items()}
    total_poss = sum(poss_ms.values())
    total_pres = sum(presence_ms.values()) or 1
    all_pids = set(by_player) | set(poss_ms) | set(presence_ms)
    for pid in all_pids:
        r = row(pid)
        ball_share = (poss_ms.get(pid, 0) / total_poss) if total_poss else 0.0
        pres_share = presence_ms.get(pid, 0) / total_pres
        # Prefer ball when we have decent coverage; otherwise lean on presence
        coverage = _ball_coverage_pct(balls, persons) / 100.0
        w_ball = 0.35 + 0.55 * coverage
        blended = w_ball * ball_share + (1 - w_ball) * pres_share
        r["possession_pct"] = round(100.0 * blended, 1)
        r["minutes"] = round(presence_ms.get(pid, 0) / 60000.0, 2)
        ball_t = int(touches.get(pid, 0))
        r["touches"] = ball_t if ball_t > 0 else int(event_touches.get(pid, 0))
    # Renormalize possession to ~100%
    poss_sum = sum(by_player[p]["possession_pct"] for p in by_player) or 1.0
    for pid in by_player:
        by_player[pid]["possession_pct"] = round(
            100.0 * by_player[pid]["possession_pct"] / poss_sum, 1
        )

    # Normalize heatmaps 0–1 for UI
    from app.services.avatars import avatar_url_for

    match_heat_n = _normalize_heat(match_heat)
    player_rows = []
    for pid, r in by_player.items():
        player = db.get(Player, UUID(pid))
        player_rows.append(
            {
                "player_id": pid,
                "display_name": player.display_name if player else "Unknown",
                "shirt_number": player.default_shirt_number if player else None,
                "team_id": str(player.team_id) if player and player.team_id else None,
                "avatar_url": avatar_url_for(player),
                "goal": r["goal"],
                "shot": r["shot"],
                "pass": r["pass"],
                "assist": r["assist"],
                "dribble": r["dribble"],
                "pass_received": r["pass_received"],
                "possession_pct": r["possession_pct"],
                "touches": r["touches"],
                "distance": round(r["distance"], 1),
                "minutes": r["minutes"],
                "avg_speed": _avg_speed(r["distance"], r["minutes"]),
                "distance_m": r["distance_m"],
                "avg_speed_kmh": r["avg_speed_kmh"],
                "top_speed_kmh": r["top_speed_kmh"],
                "conversion_pct": _conversion_pct(r["goal"], r["shot"]),
                "heatmap": _normalize_heat(r["heatmap"]),
                "possession_heatmap": _normalize_heat(r["possession_heatmap"]),
                "attribution": r["attribution"],
            }
        )

    player_rows.sort(
        key=lambda x: (-x["possession_pct"], -(x["pass"] + x["shot"] * 3 + x["goal"] * 10))
    )
    by_team = _aggregate_teams(player_rows)

    return {
        "totals": {
            **totals,
            "possession_coverage_pct": _ball_coverage_pct(balls, persons),
            "players": len(player_rows),
            "teams": len(by_team),
        },
        "by_player": player_rows,
        "by_team": by_team,
        "heatmap": {
            "width": HEAT_W,
            "height": HEAT_H,
            "match": match_heat_n,
            "match_possession": _normalize_heat(match_poss_heat),
            "unit": "image_plane_bins",
            "note": "Estimated from camera pixels — not calibrated pitch meters",
        },
        "labels": {
            "estimated": "Model estimate — not staff-confirmed",
            "confirmed": "Staff-confirmed",
            "rejected": "Rejected in review",
            "unavailable": "Metric not available for this match",
            "possession": "Share of ball-control samples attributed to this player",
            "distance": "Relative path length in image space (not metres)",
            "avg_speed": "Relative distance per minute on camera (not km/h)",
            "distance_m": "Metres run on the court (calibrated camera; foot point, jitter-filtered)",
            "avg_speed_kmh": "Average speed while on camera (calibrated camera)",
            "top_speed_kmh": "Fastest 1 s window (calibrated camera, capped at 36 km/h)",
            "minutes": "On-camera time from person tracks",
            "heatmap": "Where the player was detected most often in the camera frame",
            "possession_heatmap": "Where the player was while holding the ball (Vidal-Codina)",
            "touches": "Ball-holder changes when ball is tracked; else pass/shot involvement count",
            "dribble": "Progressive carry while retaining possession (estimated)",
            "assist": "Pass immediately preceding a goal in the possession chain (estimated)",
            "pass_received": "Completed passes where this player was the receiver",
            "conversion_pct": "Goals per shot × 100 (None if no shots)",
        },
    }


def _avg_speed(distance: float, minutes: float) -> float:
    if minutes <= 0 and distance <= 0:
        return 0.0
    return round(float(distance) / max(float(minutes), 0.01), 1)


def _conversion_pct(goals: int, shots: int) -> float | None:
    if shots <= 0:
        return None
    return round(100.0 * float(goals) / float(shots), 1)


def _aggregate_teams(player_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    for r in player_rows:
        tid = r.get("team_id") or "unknown"
        t = teams.setdefault(
            tid,
            {
                "team_id": None if tid == "unknown" else tid,
                "goal": 0,
                "shot": 0,
                "pass": 0,
                "assist": 0,
                "dribble": 0,
                "pass_received": 0,
                "touches": 0,
                "possession_pct": 0.0,
                "players": 0,
            },
        )
        for k in ("goal", "shot", "pass", "assist", "dribble", "pass_received", "touches"):
            t[k] += int(r.get(k) or 0)
        t["possession_pct"] = round(t["possession_pct"] + float(r.get("possession_pct") or 0), 1)
        t["players"] += 1
    return sorted(teams.values(), key=lambda x: (-x["goal"], -x["possession_pct"]))


def _event_type_value(et: EventType) -> str:
    return "pass" if et == EventType.pass_event else et.value


def _empty_heat() -> list[list[float]]:
    return [[0.0 for _ in range(HEAT_W)] for _ in range(HEAT_H)]


def _latest_track_players(db: Session, tracks: list[PlayerTrack]) -> dict[UUID, str]:
    out: dict[UUID, str] = {}
    for t in tracks:
        link = (
            db.query(TrackIdentityLink)
            .filter(TrackIdentityLink.track_id == t.id)
            .order_by(TrackIdentityLink.created_at.desc())
            .first()
        )
        if link and link.player_id:
            out[t.id] = str(link.player_id)
    return out


def _camera_calibrations(db: Session, match_id: UUID) -> dict[str, tuple[dict, Any]]:
    from app.models import CameraSource

    out = {}
    for cam in db.query(CameraSource).filter(CameraSource.match_id == match_id).all():
        if cam.calibration:
            try:
                out[cam.label.value] = (cam.calibration, mapper(cam.calibration))
            except Exception:  # noqa: BLE001 — a bad calibration falls back to relative units
                continue
    return out


def _center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def _path_length(series: list[dict[str, Any]]) -> float:
    if len(series) < 2:
        return 0.0
    total = 0.0
    pts = sorted(series, key=lambda s: s.get("ts_ms", 0))
    # normalize by frame diagonal so different resolutions are comparable
    w = float(pts[0].get("frame_w") or 1280)
    h = float(pts[0].get("frame_h") or 720)
    diag = (w * w + h * h) ** 0.5 or 1.0
    for a, b in zip(pts, pts[1:]):
        ax, ay = _center(a)
        bx, by = _center(b)
        total += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return total / diag * 100.0  # “relative distance units”


def _series_to_heat(series: list[dict[str, Any]]) -> list[list[float]]:
    heat = _empty_heat()
    for pt in series:
        w = float(pt.get("frame_w") or 1280)
        h = float(pt.get("frame_h") or 720)
        cx, cy = _center(pt)
        x = min(HEAT_W - 1, max(0, int((cx / w) * HEAT_W)))
        y = min(HEAT_H - 1, max(0, int((cy / h) * HEAT_H)))
        heat[y][x] += 1.0
    return heat


def _accumulate_possession_heat(
    persons: list[PlayerTrack],
    balls: list[PlayerTrack],
    track_player: dict[UUID, str],
    by_player: dict[str, dict[str, Any]],
    row_fn,
    match_poss_heat: list[list[float]],
) -> None:
    """Bin player centers only on frames where they hold the ball (PZ)."""
    ball_pts: list[dict[str, Any]] = []
    for b in balls:
        for pt in (b.features or {}).get("bbox_series") or []:
            ball_pts.append(pt)
    ball_pts.sort(key=lambda p: p.get("ts_ms", 0))
    if not ball_pts:
        return
    person_series: dict[UUID, list[dict[str, Any]]] = {}
    for t in persons:
        series = (t.features or {}).get("bbox_series") or []
        if series:
            person_series[t.id] = sorted(series, key=lambda s: s.get("ts_ms", 0))
    if not person_series:
        return
    sample_w = float(ball_pts[0].get("frame_w") or 1280)
    sample_h = float(ball_pts[0].get("frame_h") or 720)
    diag = (sample_w**2 + sample_h**2) ** 0.5
    radius = max(70.0, POSSESSION_RADIUS_FRAC * diag)
    person_lookup = TrackIndex(person_series, PERSON_TIME_WINDOW_MS)
    for bp in ball_pts:
        ts = int(bp.get("ts_ms") or 0)
        best_tid = None
        best_d = radius
        for tid in person_lookup.active(ts):
            pt = person_lookup.at(tid, ts)
            if not pt:
                continue
            d = _ball_to_player_dist(bp, pt)
            if d < best_d:
                best_d = d
                best_tid = tid
        if best_tid is None:
            continue
        pid = track_player.get(best_tid)
        ppt = person_lookup.at(best_tid, ts)
        if not ppt:
            continue
        cell = _series_to_heat([ppt])
        _add_heat(match_poss_heat, cell)
        if pid:
            r = row_fn(pid)
            _add_heat(r["possession_heatmap"], cell)


def _add_heat(dst: list[list[float]], src: list[list[float]]) -> None:
    for y in range(HEAT_H):
        for x in range(HEAT_W):
            dst[y][x] += src[y][x]


def _normalize_heat(heat: list[list[float]]) -> list[list[float]]:
    peak = max((v for row in heat for v in row), default=0.0)
    if peak <= 0:
        return heat
    return [[round(v / peak, 4) for v in row] for row in heat]


def _person_at(series: list[dict[str, Any]], ts_ms: int) -> dict[str, Any] | None:
    if not series:
        return None
    best = min(series, key=lambda s: abs(s.get("ts_ms", 0) - ts_ms))
    if abs(best.get("ts_ms", 0) - ts_ms) > PERSON_TIME_WINDOW_MS:
        return None
    return best


def _ball_to_player_dist(ball: dict[str, Any], person: dict[str, Any]) -> float:
    bx, by = _center(ball)
    x1, y1 = float(person.get("x1", 0)), float(person.get("y1", 0))
    x2, y2 = float(person.get("x2", 0)), float(person.get("y2", 0))
    pad_x = 0.15 * max(1.0, x2 - x1)
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1 = y1 + 0.45 * (y2 - y1)
    cx = min(max(bx, x1), x2)
    cy = min(max(by, y1), y2)
    return ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5


def _event_touch_counts(db: Session, events: list[MatchEvent]) -> dict[str, int]:
    """When ball timeline is empty, count primary pass/shot/goal roles as touches."""
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if e.confirmation == ConfirmationState.rejected:
            continue
        et = _event_type_value(e.event_type)
        if et not in {"pass", "shot", "goal"}:
            continue
        parts = db.query(EventParticipant).filter(EventParticipant.event_id == e.id).all()
        for p in parts:
            if not p.player_id:
                continue
            if et == "pass" and p.role not in {"passer", "primary"}:
                continue
            if et == "shot" and p.role not in {"shooter", "primary"}:
                continue
            if et == "goal" and p.role not in {"scorer", "primary"}:
                continue
            counts[str(p.player_id)] += 1
    return dict(counts)


def _possession_and_touches(
    persons: list[PlayerTrack],
    balls: list[PlayerTrack],
    track_player: dict[UUID, str],
) -> tuple[dict[str, int], dict[str, int]]:
    ball_pts: list[dict[str, Any]] = []
    for b in balls:
        for pt in (b.features or {}).get("bbox_series") or []:
            ball_pts.append(pt)
    ball_pts.sort(key=lambda p: p.get("ts_ms", 0))

    person_series: dict[UUID, list[dict[str, Any]]] = {}
    for t in persons:
        series = (t.features or {}).get("bbox_series") or []
        if series:
            person_series[t.id] = sorted(series, key=lambda s: s.get("ts_ms", 0))

    poss_ms: dict[str, int] = defaultdict(int)
    touches: dict[str, int] = defaultdict(int)
    last_holder: str | None = None

    if not ball_pts or not person_series:
        # Fallback: share "presence time" so possession UI is not empty
        for t in persons:
            pid = track_player.get(t.id)
            if not pid:
                continue
            span = max(0, t.end_ms - t.start_ms)
            poss_ms[pid] += span
        total = sum(poss_ms.values()) or 1
        # scale to look like percentages later
        return dict(poss_ms), dict(touches)

    person_lookup = TrackIndex(person_series, PERSON_TIME_WINDOW_MS)
    for i, ball in enumerate(ball_pts):
        ts = int(ball.get("ts_ms", 0))
        w = float(ball.get("frame_w") or 1280)
        h = float(ball.get("frame_h") or 720)
        diag = (w * w + h * h) ** 0.5
        radius = max(80.0, POSSESSION_RADIUS_FRAC * diag)
        best_pid = None
        best_dist = radius
        best_tid = None
        for tid in person_lookup.active(ts):
            sample = person_lookup.at(tid, ts)
            if not sample:
                continue
            dist = _ball_to_player_dist(ball, sample)
            if dist <= best_dist:
                best_dist = dist
                best_tid = tid
                best_pid = track_player.get(tid)
        if best_pid and best_tid is not None:
            # duration until next ball sample
            next_ts = int(ball_pts[i + 1].get("ts_ms", ts)) if i + 1 < len(ball_pts) else ts + 200
            poss_ms[best_pid] += max(50, next_ts - ts)
            if best_pid != last_holder:
                touches[best_pid] += 1
                last_holder = best_pid

    return dict(poss_ms), dict(touches)


def _ball_coverage_pct(balls: list[PlayerTrack], persons: list[PlayerTrack]) -> float:
    """Rough signal: how often we had ball observations vs person activity window."""
    if not persons:
        return 0.0
    p_start = min(t.start_ms for t in persons)
    p_end = max(t.end_ms for t in persons)
    window = max(1, p_end - p_start)
    ball_pts = 0
    for b in balls:
        ball_pts += len((b.features or {}).get("bbox_series") or [])
    # expect ~1 sample / 500ms if dense; report relative coverage 0-100 capped
    expected = window / 500
    return round(min(100.0, 100.0 * ball_pts / max(1.0, expected)), 1)
