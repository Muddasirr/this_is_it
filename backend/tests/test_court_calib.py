"""Court calibration: pixels → metres (gtm_pass ViewTransformer idea)."""

import pytest

from app.services.court_calib import homography, to_court, track_motion, validate_calibration

# Perspective view of a 40×20 m court: far side narrower and higher in the image
CAL = validate_calibration(
    {
        "image_points": [[100, 1000], [500, 300], [1400, 300], [1800, 1000]],
        "court_length_m": 40,
        "court_width_m": 20,
        "frame_w": 1920,
        "frame_h": 1080,
    }
)


def _box(ts, fx, fy):
    """Player box whose foot (bottom-centre) is at pixel (fx, fy)."""
    return {"ts_ms": ts, "x1": fx - 15, "x2": fx + 15, "y1": fy - 80, "y2": fy, "frame_w": 1920, "frame_h": 1080}


def test_corners_map_to_court_corners():
    H = homography(CAL)
    court = to_court(H, [[100, 1000], [500, 300], [1400, 300], [1800, 1000]])
    assert court.round(3).tolist() == [[0, 20], [0, 0], [40, 0], [40, 20]]


def test_player_running_along_near_touchline():
    # Near touchline (y=20 m) is the bottom edge: pixel x 100→1800 spans 40 m.
    # Run 10 m (425 px) in 2 s ⇒ 18 km/h.
    series = [_box(t, 100 + 425 * t / 2000, 1000) for t in range(0, 2001, 100)]
    m = track_motion(series, CAL)
    assert m["distance_m"] == pytest.approx(10.0, abs=0.3)
    assert m["top_speed_kmh"] == pytest.approx(18.0, abs=0.6)


def test_jitter_glitch_is_not_counted_as_running():
    # Standing still with one 30 m teleport (tracker swap) — must not add 60 m
    series = [_box(t, 300, 1000) for t in range(0, 3001, 100)]
    series[15] = _box(1500, 1600, 1000)
    assert track_motion(series, CAL)["distance_m"] < 1.0


def test_rejects_collinear_corners():
    with pytest.raises(ValueError):
        validate_calibration({"image_points": [[0, 0], [10, 10], [20, 20], [30, 30]]})


def test_calibrated_camera_reports_metres_in_match_stats(client, register):
    from app.core.database import get_db
    from app.main import app
    from app.models import Player, PlayerTrack, ProcessingJob, ProcessingRun, TrackIdentityLink
    from app.models.entities import CameraLabel, IdentityState, JobStage

    owner = register("calib@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "M"}).json()
    base = f"/api/v1/facilities/{fac['id']}/matches/{match['id']}"

    db = next(app.dependency_overrides[get_db]())
    from uuid import UUID

    player = Player(facility_id=UUID(fac["id"]), display_name="P1")
    job = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="calib-test")
    db.add_all([player, job])
    db.flush()
    run = ProcessingRun(job_id=job.id, stage=JobStage.detect_track)
    db.add(run)
    db.flush()
    series = [_box(t, 100 + 425 * t / 2000, 1000) for t in range(0, 2001, 100)]
    track = PlayerTrack(match_id=UUID(match["id"]), processing_run_id=run.id, camera_label=CameraLabel.A,
                        track_key="A-p1", start_ms=0,
                        end_ms=2000, confidence=0.9, features={"class": "person", "bbox_series": series})
    db.add(track)
    db.flush()
    db.add(TrackIdentityLink(track_id=track.id, player_id=player.id, state=IdentityState.proposed, confidence=0.9))
    db.commit()

    bad = client.put(f"{base}/cameras/A/calibration", headers=owner,
                     json={"image_points": [[0, 0], [1, 1], [2, 2], [3, 3]]})
    assert bad.status_code == 422
    assert client.put(f"{base}/cameras/A/calibration", headers=owner, json=CAL).status_code == 200
    assert [c["label"] for c in client.get(f"{base}/cameras", headers=owner).json()] == ["A", "B"]

    row = client.get(f"{base}/stats", headers=owner).json()["by_player"][0]
    assert row["distance_m"] == pytest.approx(10.0, abs=0.3)
    assert row["top_speed_kmh"] == pytest.approx(18.0, abs=0.6)


def test_on_court_fraction_separates_players_from_bench():
    from app.services.court_calib import on_court_fraction

    player = [_box(t, 900, 700) for t in range(0, 2000, 100)]  # mid-court
    crowd = [_box(t, 950, 200) for t in range(0, 2000, 100)]  # stands beyond the far touchline
    assert on_court_fraction(player, CAL) == 1.0
    assert on_court_fraction(crowd, CAL) == 0.0


def _soccertrack_landmarks():
    import json
    from pathlib import Path

    raw = json.loads((Path(__file__).parent / "fixtures/soccertrack_117093_keypoints.json").read_text())
    return {tuple(float(v) for v in k.strip("()").split(",")): v for k, v in raw.items()}


def _held_out_error(clicked):
    import numpy as np

    from app.services.court_calib import mapper

    pts = _soccertrack_landmarks()
    cal = validate_calibration({
        "image_points": [pts[c] for c in clicked], "court_points": [list(c) for c in clicked],
        "court_length_m": 105, "court_width_m": 68, "frame_w": 4096, "frame_h": 1080,
    })
    rest = [c for c in pts if c not in clicked]
    pred = mapper(cal)(np.array([pts[c] for c in rest]))
    return float(np.linalg.norm(pred - np.array(rest), axis=1).mean()), cal


def test_wide_camera_needs_distortion_model_real_soccertrack_landmarks():
    corners = [(0, 68), (0, 0), (105, 0), (105, 68)]
    eight = corners + [(52.5, 0), (52.5, 68), (52.5, 24.85), (52.5, 43.15)]
    err4, cal4 = _held_out_error(corners)
    err8, cal8 = _held_out_error(eight)
    assert cal4["model"] == "homography" and cal8["model"] == "radial_homography"
    assert err4 > 5.0  # flat 4-corner transform cannot map a panorama
    assert err8 < 1.5  # measured 0.93 m
