"""Per-player stats matching CourtLens match_stats logic (no DB)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from identity_offline import PlayerCluster, TrackView
from vision import EventCandidate

POSSESSION_RADIUS_FRAC = 0.18
PERSON_TIME_WINDOW_MS = 2500
HEAT_W, HEAT_H = 24, 16


def build_player_stats(
    tracks: list[TrackView],
    players: list[PlayerCluster],
    track_to_player: dict[str, str],
    events: list[EventCandidate],
) -> dict[str, Any]:
    by_player: dict[str, dict[str, Any]] = {}
    player_meta = {p.player_key: p for p in players}

    def row(pid: str) -> dict[str, Any]:
        meta = player_meta.get(pid)
        return by_player.setdefault(
            pid,
            {
                "player_id": pid,
                "display_name": meta.display_name if meta else pid,
                "shirt_number": meta.shirt_number if meta else None,
                "kit": meta.kit if meta else None,
                "avatar_path": meta.thumb_path if meta else None,
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
                "conversion_pct": None,
                "heatmap": _empty_heat(),
                "possession_heatmap": _empty_heat(),
                "attribution": "estimated",
            },
        )

    totals = {"goal": 0, "shot": 0, "pass": 0, "assist": 0, "dribble": 0}
    for e in events:
        et = e.event_type
        if et in totals:
            totals[et] += 1
        keys = e.track_keys or []
        primary = keys[0] if keys else None
        if et == "pass":
            if primary and primary in track_to_player:
                row(track_to_player[primary])["pass"] += 1
            if len(keys) > 1 and keys[1] in track_to_player:
                row(track_to_player[keys[1]])["pass_received"] += 1
        elif et == "shot" and primary and primary in track_to_player:
            row(track_to_player[primary])["shot"] += 1
        elif et == "goal" and primary and primary in track_to_player:
            row(track_to_player[primary])["goal"] += 1
        elif et == "assist" and primary and primary in track_to_player:
            row(track_to_player[primary])["assist"] += 1
        elif et == "dribble" and primary and primary in track_to_player:
            row(track_to_player[primary])["dribble"] += 1

    persons = [t for t in tracks if t.class_name == "person"]
    balls = [t for t in tracks if t.class_name == "sports_ball"]

    match_heat = _empty_heat()
    for t in persons:
        pid = track_to_player.get(t.track_key)
        series = t.bbox_series or (t.features or {}).get("bbox_series") or []
        dist = _path_length(series)
        heat = _series_to_heat(series)
        _add_heat(match_heat, heat)
        if pid:
            r = row(pid)
            r["distance"] += dist
            _add_heat(r["heatmap"], heat)

    poss_ms, touches = _possession_and_touches(persons, balls, track_to_player)
    presence_ms: dict[str, int] = defaultdict(int)
    for t in persons:
        pid = track_to_player.get(t.track_key)
        if pid:
            presence_ms[pid] += max(0, t.end_ms - t.start_ms)

    total_poss = sum(poss_ms.values())
    total_pres = sum(presence_ms.values()) or 1
    coverage = _ball_coverage_pct(balls, persons) / 100.0
    w_ball = 0.35 + 0.55 * coverage
    all_pids = set(by_player) | set(poss_ms) | set(presence_ms) | set(player_meta)
    event_touches = _event_touch_counts(events, track_to_player)
    for pid in all_pids:
        r = row(pid)
        ball_share = (poss_ms.get(pid, 0) / total_poss) if total_poss else 0.0
        pres_share = presence_ms.get(pid, 0) / total_pres
        blended = w_ball * ball_share + (1 - w_ball) * pres_share
        r["possession_pct"] = round(100.0 * blended, 1)
        r["minutes"] = round(presence_ms.get(pid, 0) / 60000.0, 2)
        # Prefer ball-holder touches; fall back to pass/shot involvement when ball is sparse
        ball_t = int(touches.get(pid, 0))
        r["touches"] = ball_t if ball_t > 0 else int(event_touches.get(pid, 0))

    poss_sum = sum(by_player[p]["possession_pct"] for p in by_player) or 1.0
    for pid in by_player:
        by_player[pid]["possession_pct"] = round(
            100.0 * by_player[pid]["possession_pct"] / poss_sum, 1
        )
        by_player[pid]["distance"] = round(by_player[pid]["distance"], 2)
        by_player[pid]["avg_speed"] = round(
            by_player[pid]["distance"] / max(by_player[pid]["minutes"], 0.01), 1
        )
        by_player[pid]["conversion_pct"] = (
            round(100.0 * by_player[pid]["goal"] / by_player[pid]["shot"], 1)
            if by_player[pid]["shot"]
            else None
        )
        by_player[pid]["heatmap"] = _normalize_heat(by_player[pid]["heatmap"])

    player_rows = sorted(
        by_player.values(),
        key=lambda r: (-(r["touches"] + r["shot"] + r["pass"]), -r["possession_pct"]),
    )
    return {
        "totals": totals,
        "ball_coverage_pct": _ball_coverage_pct(balls, persons),
        "match_heatmap": _normalize_heat(match_heat),
        "players": player_rows,
    }


def _event_touch_counts(
    events: list[EventCandidate], track_to_player: dict[str, str]
) -> dict[str, int]:
    """Count a touch per pass/shot/goal primary participant when ball timeline is empty."""
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if e.event_type not in {"pass", "shot", "goal"}:
            continue
        keys = e.track_keys or []
        if not keys:
            continue
        pid = track_to_player.get(keys[0])
        if pid:
            counts[pid] += 1
    return dict(counts)


def _empty_heat() -> list[list[float]]:
    return [[0.0] * HEAT_W for _ in range(HEAT_H)]


def _center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def _path_length(series: list[dict[str, Any]]) -> float:
    if len(series) < 2:
        return 0.0
    total = 0.0
    pts = sorted(series, key=lambda s: s.get("ts_ms", 0))
    w = float(pts[0].get("frame_w") or 1280)
    h = float(pts[0].get("frame_h") or 720)
    diag = (w * w + h * h) ** 0.5 or 1.0
    for a, b in zip(pts, pts[1:]):
        ax, ay = _center(a)
        bx, by = _center(b)
        total += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return total / diag * 100.0


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


def _possession_and_touches(
    persons: list[TrackView],
    balls: list[TrackView],
    track_player: dict[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    ball_pts: list[dict[str, Any]] = []
    for b in balls:
        for pt in b.bbox_series or (b.features or {}).get("bbox_series") or []:
            ball_pts.append(pt)
    ball_pts.sort(key=lambda p: p.get("ts_ms", 0))

    person_series: dict[str, list[dict[str, Any]]] = {}
    for t in persons:
        series = t.bbox_series or (t.features or {}).get("bbox_series") or []
        if series:
            person_series[t.track_key] = sorted(series, key=lambda s: s.get("ts_ms", 0))

    poss_ms: dict[str, int] = defaultdict(int)
    touches: dict[str, int] = defaultdict(int)
    last_holder: str | None = None

    if not ball_pts or not person_series:
        for t in persons:
            pid = track_player.get(t.track_key)
            if pid:
                poss_ms[pid] += max(0, t.end_ms - t.start_ms)
        return dict(poss_ms), dict(touches)

    for i, ball in enumerate(ball_pts):
        ts = int(ball.get("ts_ms", 0))
        w = float(ball.get("frame_w") or 1280)
        h = float(ball.get("frame_h") or 720)
        diag = (w * w + h * h) ** 0.5
        radius = max(80.0, POSSESSION_RADIUS_FRAC * diag)
        best_pid = None
        best_dist = radius
        for tid, series in person_series.items():
            sample = _person_at(series, ts)
            if not sample:
                continue
            dist = _ball_to_player_dist(ball, sample)
            if dist <= best_dist:
                best_dist = dist
                best_pid = track_player.get(tid)
        if best_pid:
            next_ts = int(ball_pts[i + 1].get("ts_ms", ts)) if i + 1 < len(ball_pts) else ts + 200
            poss_ms[best_pid] += max(50, next_ts - ts)
            if best_pid != last_holder:
                touches[best_pid] += 1
                last_holder = best_pid

    return dict(poss_ms), dict(touches)


def _ball_coverage_pct(balls: list[TrackView], persons: list[TrackView]) -> float:
    if not persons:
        return 0.0
    p_start = min(t.start_ms for t in persons)
    p_end = max(t.end_ms for t in persons)
    window = max(1, p_end - p_start)
    ball_pts = 0
    for b in balls:
        ball_pts += len(b.bbox_series or (b.features or {}).get("bbox_series") or [])
    expected = window / 500
    return round(min(100.0, 100.0 * ball_pts / max(1.0, expected)), 1)
