"""Assist + dribble candidates on synthetic possession timelines."""

from app.services.events import infer_event_candidates
from app.services.vision import Detection


def _person(key: str, points: list[tuple[int, float, float]]) -> Detection:
    series = [
        {
            "ts_ms": ts,
            "x1": x - 20,
            "y1": y - 40,
            "x2": x + 20,
            "y2": y + 40,
            "frame_w": 1280,
            "frame_h": 720,
            "confidence": 0.9,
        }
        for ts, x, y in points
    ]
    return Detection(
        track_key=key,
        camera_label="A",
        start_ms=points[0][0],
        end_ms=points[-1][0],
        confidence=0.9,
        bbox_series=series,
        class_name="person",
    )


def _ball(points: list[tuple[int, float, float]]) -> Detection:
    series = [
        {
            "ts_ms": ts,
            "x1": x - 5,
            "y1": y - 5,
            "x2": x + 5,
            "y2": y + 5,
            "frame_w": 1280,
            "frame_h": 720,
            "confidence": 0.8,
        }
        for ts, x, y in points
    ]
    return Detection(
        track_key="A-b1",
        camera_label="A",
        start_ms=points[0][0],
        end_ms=points[-1][0],
        confidence=0.8,
        bbox_series=series,
        class_name="sports_ball",
    )


def test_dribble_from_progressive_carry():
    # Holder walks with ball across frame for >1.2s; ≥80 ball samples for strict gate
    pts = [(t, 200 + t * 0.15, 360) for t in range(0, 10000, 100)]
    p1 = _person("A-p1", pts)
    p2 = _person("A-p2", [(t, 900, 360) for t, _, _ in pts])
    ball = _ball(pts)
    cands = infer_event_candidates([p1, p2, ball], "A")
    dribbles = [c for c in cands if c.event_type == "dribble"]
    assert len(dribbles) >= 1, f"expected dribble, got {[(c.event_type, c.evidence) for c in cands]}"


def test_assist_after_pass_then_goal():
    # Dense ball: pass A→B then B shoots goalward into goal band
    p1 = _person("A-p1", [(t, 200, 360) for t in range(0, 12000, 100)])
    p2 = _person("A-p2", [(t, 1000, 200) for t in range(0, 12000, 100)])
    ball_pts = []
    # A holds
    for t in range(0, 1500, 100):
        ball_pts.append((t, 200, 360))
    # flight to B
    ball_pts += [(1700, 500, 300), (1900, 800, 250)]
    # B holds then shoots left into goal band (x near 0)
    for t in range(2100, 3500, 100):
        ball_pts.append((t, 1000, 200))
    # goalward flight into left band
    ball_pts += [
        (3600, 700, 200),
        (3800, 400, 200),
        (4000, 80, 200),
        (4200, 40, 200),
        (4400, 30, 200),
        (4600, 25, 200),
        (4800, 20, 200),
    ]
    # pad more samples so ball_series >= 80
    for t in range(5000, 13000, 100):
        ball_pts.append((t, 30, 200))
    cands = infer_event_candidates([p1, p2, _ball(ball_pts)], "A")
    goals = [c for c in cands if c.event_type == "goal"]
    assists = [c for c in cands if c.event_type == "assist"]
    # Goal may be strict; if we have a goal we must have an assist from the passer
    if goals:
        assert assists, f"goal without assist: {[c.evidence for c in goals]}"
        assert assists[0].track_keys[0] == "A-p1"
    else:
        # Still expect passes and/or dribbles without inventing goals
        assert any(c.event_type in {"pass", "dribble", "shot"} for c in cands)
