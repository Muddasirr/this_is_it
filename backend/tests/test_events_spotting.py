from app.services.events import events_from_spotting
from app.services.vision import Detection


def _person(key, x, t0, t1):
    series = [{"ts_ms": t, "x1": x, "y1": 400, "x2": x + 40, "y2": 520, "frame_w": 1280, "frame_h": 720}
              for t in range(t0, t1 + 1, 100)]
    return Detection(track_key=key, camera_label="A", start_ms=t0, end_ms=t1, confidence=0.9,
                     bbox_series=series, class_name="person")


def _ball(points):
    series = [{"ts_ms": t, "x1": x - 5, "y1": 510, "x2": x + 5, "y2": 520, "frame_w": 1280, "frame_h": 720}
              for t, x in points]
    return Detection(track_key="A-b0", camera_label="A", start_ms=points[0][0], end_ms=points[-1][0],
                     confidence=0.5, bbox_series=series, class_name="sports_ball")


def test_spotted_pass_gets_passer_and_receiver_from_possession():
    # ball at A's feet 0–2 s, travels, at B's feet 2.6–5 s
    pts = [(t, 120) for t in range(0, 2001, 100)] + [(t, 120 + (t - 2000) * 1.0) for t in range(2100, 2600, 100)]
    pts += [(t, 620) for t in range(2600, 5001, 100)]
    dets = [_person("A-p1", 100, 0, 5000), _person("A-p2", 600, 0, 5000), _ball(pts)]
    out = events_from_spotting([{"ts_ms": 2050, "event_type": "pass", "label": "PASS", "confidence": 0.8}], dets, "A")
    assert len(out) == 1 and out[0].event_type == "pass"
    assert out[0].track_keys == ["A-p1", "A-p2"]
    assert out[0].evidence["method"] == "tdeed_soccertrack"


def test_event_without_a_holder_stays_unattributed():
    dets = [_person("A-p1", 100, 0, 1000), _ball([(t, 900) for t in range(0, 1001, 100)])]  # ball far from anyone
    out = events_from_spotting([{"ts_ms": 500, "event_type": "shot", "label": "SHOT", "confidence": 0.6}], dets, "A")
    assert len(out) == 1 and out[0].track_keys == []


def test_pass_to_the_other_team_becomes_an_interception():
    from types import SimpleNamespace

    from app.workers.pipeline import drop_self_passes

    def p(passer, receiver):
        return SimpleNamespace(event_type="pass", evidence={"passer": passer, "receiver": receiver})

    player_of = {"t1": "P1", "t2": "P2", "t3": "P3", "t4": "P1"}
    team_of = {"P1": "red", "P2": "red", "P3": "black"}
    kept = drop_self_passes([p("t1", "t2"), p("t1", "t3"), p("t1", "t4")], player_of, team_of)
    assert [(c.event_type, c.evidence["passer"], c.evidence["receiver"]) for c in kept] == [
        ("pass", "t1", "t2"), ("interception", "t1", "t3")]
    assert kept[1].evidence["from_pass"] is True


def test_spot_nms_keeps_the_strongest_of_a_burst():
    from app.services.events import _spot_nms

    spots = [{"ts_ms": 25000, "event_type": "shot", "confidence": 0.16},
             {"ts_ms": 25600, "event_type": "shot", "confidence": 0.9},
             {"ts_ms": 25300, "event_type": "pass", "confidence": 0.5},  # other type: kept
             {"ts_ms": 29600, "event_type": "shot", "confidence": 0.54}]
    kept = sorted((e["ts_ms"], e["event_type"]) for e in _spot_nms(spots))
    assert kept == [(25300, "pass"), (25600, "shot"), (29600, "shot")]


def test_football_format_uses_the_football_model_and_thresholds(monkeypatch):
    from app.services import action_spotting

    monkeypatch.setenv("TDEED_WEIGHTS", "/m/tdeed_futsal.pt")
    monkeypatch.setenv("TDEED_WEIGHTS_FOOTBALL", "/m/tdeed_soccertrack.pt")
    monkeypatch.setenv("COURT_FORMAT", "11v11")
    assert action_spotting.weights_path() == "/m/tdeed_soccertrack.pt"
    assert action_spotting.thresholds() == action_spotting.FOOTBALL_THRESHOLDS
    monkeypatch.setenv("COURT_FORMAT", "5v5")
    assert action_spotting.weights_path() == "/m/tdeed_futsal.pt"
    assert action_spotting.thresholds() == action_spotting.THRESHOLDS
