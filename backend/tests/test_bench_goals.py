from app.scripts.bench_goals import score


def test_goal_matched_in_lead_window_one_to_one():
    gt = [{"ts_ms": 60000, "type": "goal"}, {"ts_ms": 120000, "type": "goal"}]
    pred = [
        {"ts_ms": 50000, "event_type": "shot"},
        {"ts_ms": 55000, "event_type": "goal"},
        {"ts_ms": 56000, "event_type": "goal"},  # duplicate, only one may match
        {"ts_ms": 130000, "event_type": "goal"},  # after the lag window
    ]
    r = score(gt, pred)
    assert r["goal_recall"] == 0.5
    assert r["goal_precision"] == round(1 / 3, 3)
    assert r["goal_shot_recall"] == 0.5


def test_no_predictions():
    r = score([{"ts_ms": 1000, "type": "goal"}], [])
    assert r["goal_recall"] == 0.0 and r["goal_precision"] is None
