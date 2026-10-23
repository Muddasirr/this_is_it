"""Unit tests for possession-centric pass detection (no YOLO)."""

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


def test_possession_change_creates_multiple_passes():
    # Player A holds ball near (200,360), then ball flies to B at (800,360)
    # Repeat pattern three times → expect multiple passes (not capped at 8)
    persons = []
    ball_pts = []
    for i in range(6):
        base = i * 3000
        ax, ay = 200.0, 360.0
        bx, by = 800.0, 360.0
        if i % 2 == 0:
            holder_a = [(base + t, ax, ay) for t in (0, 200, 400, 600)]
            holder_b = [(base + 1500 + t, bx, by) for t in (0, 200, 400)]
            ball_pts.extend([(base + t, ax, ay) for t in (0, 200, 400, 600)])
            ball_pts.extend([(base + 900, 500, 360), (base + 1200, 650, 360)])
            ball_pts.extend([(base + 1500 + t, bx, by) for t in (0, 200, 400)])
        else:
            holder_a = [(base + 1500 + t, ax, ay) for t in (0, 200, 400)]
            holder_b = [(base + t, bx, by) for t in (0, 200, 400, 600)]
            ball_pts.extend([(base + t, bx, by) for t in (0, 200, 400, 600)])
            ball_pts.extend([(base + 900, 500, 360), (base + 1200, 350, 360)])
            ball_pts.extend([(base + 1500 + t, ax, ay) for t in (0, 200, 400)])
        persons.append(_person("A-p1", holder_a if i % 2 == 0 else [(base + t, ax, ay) for t in (0, 200, 400, 600, 1500, 1700, 1900)]))
        persons.append(_person("A-p2", holder_b if i % 2 == 0 else [(base + t, bx, by) for t in (0, 200, 400, 600, 1500, 1700, 1900)]))

    # Simpler clean sequence: 5 A→B possession changes
    p1 = _person(
        "A-p1",
        [(t, 200, 360) for t in range(0, 20000, 200)],
    )
    p2 = _person(
        "A-p2",
        [(t, 900, 360) for t in range(0, 20000, 200)],
    )
    ball = []
    for cycle in range(5):
        t0 = cycle * 3500
        # A possession
        for t in range(0, 1000, 200):
            ball.append((t0 + t, 200, 360))
        # flight
        ball.append((t0 + 1200, 450, 360))
        ball.append((t0 + 1400, 700, 360))
        # B possession
        for t in range(1600, 2600, 200):
            ball.append((t0 + t, 900, 360))

    # Pad dense ball so len(ball_series) >= MIN_BALL_FOR_STRICT (80)
    while len(ball) < 90:
        t_last = ball[-1][0] + 100
        ball.append((t_last, ball[-1][1], ball[-1][2]))
    dets = [p1, p2, _ball(ball)]
    cands = infer_event_candidates(dets, "A")
    passes = [c for c in cands if c.event_type == "pass"]
    assert len(passes) >= 4, f"expected several passes, got {len(passes)}: {passes}"


def test_short_dense_clip_is_not_rejected_by_absolute_ball_gate():
    """A 2.5 s labelled clip has ≤ 63 ball samples; the old ≥ 80 gate returned [] always."""
    from app.services.events import _enough_ball

    dense_clip = [{"ts_ms": t} for t in range(0, 2500, 80)]  # 32 samples, 12.5/s
    sparse = [{"ts_ms": t} for t in range(0, 60_000, 2000)]  # 30 samples, 0.5/s
    assert _enough_ball(dense_clip)
    assert not _enough_ball(sparse)
