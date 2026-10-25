"""Scoreboard → goals: lasting increments only; flicker and OCR jumps ignored."""

from app.scripts.scoreboard_goals import goals_from_series


def _s(scores):
    return [{"ts_ms": i * 500, "score": s, "clock": None} for i, s in enumerate(scores)]


def test_goals_from_lasting_score_changes():
    series = _s([(0, 0)] * 4 + [None] * 3 + [(1, 0)] * 4 + [(7, 0)] + [(1, 0)] * 2 + [(1, 1)] * 3)
    goals = goals_from_series(series, hold=3)
    assert [(g["side"], g["score_after"]) for g in goals] == [("home", [1, 0]), ("away", [1, 1])]
    assert goals[0]["ts_ms"] == 7 * 500  # first frame showing 1-0


def test_flicker_shorter_than_hold_is_not_a_goal():
    assert goals_from_series(_s([(2, 2)] * 4 + [(2, 3)] * 2 + [(2, 2)] * 4), hold=3) == []
