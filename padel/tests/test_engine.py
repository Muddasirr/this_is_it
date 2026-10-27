import numpy as np
import pytest
from conftest import synthetic_detections

from padel_analyzer.engine import ball as B
from padel_analyzer.engine import court as C
from padel_analyzer.engine import pipeline
from padel_analyzer.engine import players as P
from padel_analyzer.engine import rallies as R
from padel_analyzer.engine.events import Hit
from padel_analyzer.eval import metrics as M


def test_landmarks_follow_fip_geometry():
    assert C.LANDMARKS["far_service_t"] == (5.0, 3.05)
    assert C.LANDMARKS["near_service_left"][1] - C.NET_Y == pytest.approx(6.95)
    assert list(C.zone([10.0, 6.5, 5.0, 2.0, 17.5])) == [0, 0, 1, 2, 2]


def test_fit_recovers_homography(calib):
    pts = {k: tuple(v) for k, v in zip(C.LANDMARKS, calib.to_px(np.array(list(C.LANDMARKS.values()))))}
    est = C.fit(pts, (1920, 1080))
    assert C.floor_error_m(est.H, calib.H) < 0.01
    four = {k: pts[k] for k in ["far_service_left", "far_service_right", "near_service_left", "near_service_right"]}
    assert C.floor_error_m(C.fit(four, (1920, 1080)).H, calib.H) < 0.01


def test_fit_rejects_collinear(calib):
    pts = {k: tuple(v) for k, v in zip(C.LANDMARKS, calib.to_px(np.array(list(C.LANDMARKS.values()))))}
    line = {k: pts[k] for k in ["far_service_left", "far_service_t", "far_service_right", "far_left"]}
    line["far_left"] = pts["far_service_left"]
    with pytest.raises(ValueError):
        C.fit({k: pts[k] for k in ["net_left", "net_centre", "net_right"]}, (1920, 1080))


def test_calibration_roundtrip(calib):
    d = calib.to_dict()
    back = C.Calibration.from_dict(d)
    assert np.allclose(back.H, calib.H)


def test_fit_snaps_onto_lines_is_safe(calib):
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:] = (160, 60, 20)
    for a, b in C.PAINTED:
        p = calib.to_px(np.array([a, b])).astype(int)
        import cv2

        cv2.line(img, tuple(p[0]), tuple(p[1]), (255, 255, 255), 4)
    pts = {k: tuple(v + np.array([7, -6])) for k, v in
           zip(C.LANDMARKS, calib.to_px(np.array(list(C.LANDMARKS.values()))))}
    rough = C.fit({k: pts[k] for k in ["far_service_left", "far_service_right", "near_service_left",
                                       "near_service_right"]}, (1920, 1080))
    snapped, info = C.refine(img, rough)
    assert info["line_cost_after_px"] <= info["line_cost_before_px"]
    assert C.floor_error_m(snapped.H, calib.H) <= C.floor_error_m(rough.H, calib.H) + 1e-6


def test_ball_linker_rejects_static_lookalike(calib):
    det, _ = synthetic_detections(calib)
    bt = B.track(det.ball)
    t = np.arange(det.n) / det.fps
    gt = np.c_[960 + 300 * np.sin(t * 2), 500 + 250 * np.sin(t * 3)]
    pr = M.ball_pr(bt.xy, gt, tol_px=10)
    assert pr["recall"] > 0.95 and pr["precision"] > 0.95
    assert not np.any(np.linalg.norm(bt.xy - [104, 904], axis=1) < 5)


def test_player_tracking_positions(calib):
    det, pos = synthetic_detections(calib)
    tr = P.track(det, calib)
    err = M.position_error(tr.pos_m, pos)
    assert err["median_m"] < 0.05 and err["coverage"] > 0.99
    # slots keep their side of the net
    assert np.nanmax(tr.pos_m[:, :2, 1]) < 10 and np.nanmin(tr.pos_m[:, 2:, 1]) > 10


def test_cut_starts_new_segment(calib):
    det, pos = synthetic_detections(calib, n=240, cut_at=120)
    tr = P.track(det, calib)
    assert len(tr.segments) == 2 and tr.segments[1][0] == 120
    assert M.position_error(tr.pos_m, pos)["median_m"] < 0.05


def test_rally_grouping_splits_on_gap_and_cut():
    hits = [Hit(f, 0, 1.0) for f in (10, 40, 70, 300, 330, 600)]
    r = R.group(hits, 30.0, 900, [(0, 500), (500, 900)])
    assert [x.hits for x in r] == [[0, 1, 2], [3, 4], [5]]


def test_dead_ball_touches_are_not_rallies():
    r = R.Rally(0, 0, 10, hits=[0, 1, 2])
    assert R.dead_ball(r, [0, 0, 0])  # one pair knocking the ball back to the server
    assert not R.dead_ball(r, [0, 1, 0])  # an exchange is a point
    assert not R.dead_ball(R.Rally(0, 0, 10, hits=[0, 1, 2, 3]), [1, 1, 1, 1])  # long one-team runs: keep


def test_analyze_report_shape(calib):
    det, _ = synthetic_detections(calib, n=300)
    rep = pipeline.analyze(det, calib)
    assert set(rep) >= {"match", "players", "teams", "rallies", "shots", "quality", "replay"}
    assert len(rep["players"]) == 4 and len(rep["teams"]) == 2
    for p in rep["players"]:
        assert p["stats"]["distance_m"] > 0
        assert abs(sum(p["stats"]["zone_pct"].values()) - 100) < 0.5
    assert rep["quality"]["players_tracked_pct"] > 95
    assert len(rep["replay"]["players"]) == 50


def test_outcome_stats_replays_padel_scoring():
    from padel_analyzer.engine import advanced as ADV

    # game 1: team 0 serves, wins 4-0; game 2: team 1 serves, 3-3 then 0 wins two (advantage) -> break
    winners = [0, 0, 0, 0] + [1, 0, 1, 0, 1, 0, 0, 0]
    servers = [0] * 4 + [1] * 8
    shots, rallies = [], []
    for i, (w, s) in enumerate(zip(winners, servers)):
        shots += [{"player": 2 * s, "team": s}, {"player": 2 * (1 - s), "team": 1 - s}]
        rallies.append({"id": i, "shots": [2 * i, 2 * i + 1], "winner_team": w, "winner_source": "confirmed"})
    rep = {"shots": shots, "rallies": rallies, "match": {},
           "players": [{"id": p, "team": 0 if p < 2 else 1, "stats": {}} for p in range(4)],
           "teams": [{"id": t, "stats": {}} for t in (0, 1)]}
    ADV.outcome_stats(rep)
    assert rep["match"]["score"]["sets"] == [[2, 0]]
    assert [g["server_team"] for g in rep["match"]["games"]] == [0, 1]
    assert rep["teams"][0]["stats"]["breaks"] == 1 and rep["teams"][0]["stats"]["service_games_held"] == 1
    assert rep["teams"][0]["stats"]["points_won"] == 9
    assert rep["teams"][0]["stats"]["serve_points_won_pct"] == 100.0
