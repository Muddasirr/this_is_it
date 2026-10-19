"""Phase 3 event inference — heuristic candidates from tracks (pretrained detect/track only)."""

from __future__ import annotations

from typing import Any

from app.services.vision import Detection, EventCandidate


def infer_event_candidates(
    detections: list[Detection],
    camera_label: str,
) -> list[EventCandidate]:
    """Generate low-confidence goal/shot/pass candidates for staff review.

    Not a trained event model. Uses motion heuristics so the review workflow
    can be exercised before Phase 3 quality gates are locked.
    """
    persons = [d for d in detections if d.class_name == "person"]
    balls = [d for d in detections if d.class_name == "sports_ball"]
    candidates: list[EventCandidate] = []

    # Ball near top/bottom of frame → goal candidate (court orientation unknown)
    for ball in balls:
        series = ball.bbox_series or []
        for pt in series:
            y1, y2 = pt.get("y1", 0), pt.get("y2", 0)
            # normalized later if we lack frame size; use absolute extremes as weak signal
            cy = (y1 + y2) / 2
            if cy < 80 or (pt.get("frame_h") and cy > 0.9 * pt["frame_h"]):
                candidates.append(
                    EventCandidate(
                        event_type="goal",
                        camera_label=camera_label,
                        camera_ts_ms=pt.get("ts_ms", ball.start_ms),
                        confidence=0.35,
                        track_keys=[ball.track_key] + [p.track_key for p in _nearby(persons, pt, 200)],
                        evidence={"heuristic": "ball_near_vertical_extreme", "point": pt},
                    )
                )
                break

    # Person track with sudden bbox jump → shot-like motion
    for person in persons:
        series = person.bbox_series or []
        if len(series) < 2:
            continue
        for a, b in zip(series, series[1:]):
            speed = _center_delta(a, b)
            dt = max(1, b.get("ts_ms", 0) - a.get("ts_ms", 0))
            if speed / dt > 0.4:  # px/ms threshold — loose for review queue
                candidates.append(
                    EventCandidate(
                        event_type="shot",
                        camera_label=camera_label,
                        camera_ts_ms=b.get("ts_ms", person.start_ms),
                        confidence=0.4,
                        track_keys=[person.track_key],
                        evidence={"heuristic": "rapid_motion", "speed_px_per_ms": speed / dt},
                    )
                )
                break

    # Two person tracks overlapping in time → limited pass candidate
    for i, a in enumerate(persons):
        for b in persons[i + 1 :]:
            overlap_start = max(a.start_ms, b.start_ms)
            overlap_end = min(a.end_ms, b.end_ms)
            if overlap_end - overlap_start < 400:
                continue
            mid = (overlap_start + overlap_end) // 2
            candidates.append(
                EventCandidate(
                    event_type="pass",
                    camera_label=camera_label,
                    camera_ts_ms=mid,
                    confidence=0.3,
                    track_keys=[a.track_key, b.track_key],
                    evidence={"heuristic": "coincident_tracks", "overlap_ms": overlap_end - overlap_start},
                )
            )
            if len([c for c in candidates if c.event_type == "pass"]) >= 8:
                break
        if len([c for c in candidates if c.event_type == "pass"]) >= 8:
            break

    return _dedupe(candidates)


def _center(pt: dict[str, Any]) -> tuple[float, float]:
    return ((pt.get("x1", 0) + pt.get("x2", 0)) / 2, (pt.get("y1", 0) + pt.get("y2", 0)) / 2)


def _center_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _nearby(persons: list[Detection], pt: dict[str, Any], radius: float) -> list[Detection]:
    px, py = _center(pt)
    out = []
    for p in persons:
        series = p.bbox_series or []
        if not series:
            continue
        # pick closest-in-time sample
        sample = min(series, key=lambda s: abs(s.get("ts_ms", 0) - pt.get("ts_ms", 0)))
        cx, cy = _center(sample)
        if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 <= radius:
            out.append(p)
    return out[:2]


def _dedupe(candidates: list[EventCandidate]) -> list[EventCandidate]:
    seen: set[tuple] = set()
    out: list[EventCandidate] = []
    for c in candidates:
        key = (c.event_type, c.camera_ts_ms // 1500, tuple(sorted(c.track_keys)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
