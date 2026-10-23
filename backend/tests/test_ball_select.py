"""Match-ball selection: spare balls by the touchline must not win."""

from app.services.ball_select import select_ball, static_cells


def _spare(ts):  # big, sharp, never moves
    return (ts, 1000, 900, 1020, 920, 0.6)


def _match(ts):  # small, blurrier, moving across the pitch
    x = 300 + ts * 0.05
    return (ts, x, 500, x + 14, 514, 0.35)


def test_static_spare_ball_is_suppressed_and_moving_ball_chosen():
    ts_all = list(range(0, 60_000, 100))  # 60 s at 10 Hz
    cands = [_spare(t) for t in ts_all] + [_match(t) for t in ts_all]
    assert static_cells(cands, len(ts_all))  # spare ball cell found
    chosen = select_ball(cands, len(ts_all))
    assert len(chosen) == len(ts_all)
    assert all(c[2] != 900 for c in chosen)  # never the spare ball (it sits at y = 900)


def test_trajectory_beats_confidence_for_a_one_frame_false_positive():
    ts_all = list(range(0, 5_000, 100))
    cands = [_match(t) for t in ts_all] + [(2_000, 3000, 200, 3014, 214, 0.9)]  # far-away flash
    chosen = {c[0]: c for c in select_ball(cands, len(ts_all))}
    assert chosen[2_000][1] != 3000
