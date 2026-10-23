"""SoccerTrack v2 event scorer: class mapping, tolerance matching, team-split accuracy."""

import json

from app.scripts.bench_soccertrack import load_pred, load_truth, score


def _truth_file(tmp_path):
    actions = [
        {"gameTime": "1 - 0:01", "label": "PASS", "position": "1000", "frame": 25, "team": "left"},
        {"gameTime": "1 - 0:03", "label": "HIGH PASS", "position": "3000", "frame": 75, "team": "right"},
        {"gameTime": "1 - 0:05", "label": "SHOT", "position": "5000", "frame": 125, "team": "left"},
        {"gameTime": "1 - 0:06", "label": "OUT", "position": "6000", "frame": 150, "team": "left"},
        {"gameTime": "2 - 0:01", "label": "PASS", "position": "1000", "frame": 25, "team": "left"},
    ]
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"match_id": 1, "fps": 25.0, "actions": actions}))
    return p


def _run_dir(tmp_path, events):
    d = tmp_path / "run"
    d.mkdir()
    (d / "canonical_events.json").write_text(json.dumps(events))
    return d


def test_scores_passes_shots_and_team_split(tmp_path):
    truth = load_truth(_truth_file(tmp_path), half=1)
    assert [t["type"] for t in truth] == ["pass", "pass", "shot"]  # OUT ignored, half 2 excluded
    pred = load_pred(_run_dir(tmp_path, [
        {"event_type": "pass", "ts_ms": 1200, "participants": [{"role": "passer", "player": "Kit B #4"}]},
        {"event_type": "pass", "ts_ms": 2900, "participants": [{"role": "passer", "player": "Kit A #9"}]},
        {"event_type": "pass", "ts_ms": 9000, "participants": []},  # false positive
        {"event_type": "shot", "ts_ms": 7000, "participants": []},  # outside ±1 s
    ]))
    s = score(truth, pred, tol_ms=1000)
    assert s["by_class"]["pass"] == {"truth": 2, "pred": 3, "tp": 2, "precision": 0.667, "recall": 1.0, "f1": 0.8}
    assert s["by_class"]["shot"]["tp"] == 0
    # left→B, right→A consistently: the flipped mapping scores 100 %
    assert s["team_split"]["accuracy"] == 1.0
