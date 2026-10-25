"""Pure helpers for per-player / per-team stats (no DB)."""

from app.services.match_stats import _aggregate_teams, _avg_speed, _conversion_pct, _path_length


def test_conversion_pct():
    assert _conversion_pct(0, 0) is None
    assert _conversion_pct(1, 4) == 25.0
    assert _conversion_pct(2, 2) == 100.0


def test_avg_speed_relative():
    assert _avg_speed(0, 0) == 0.0
    assert _avg_speed(10, 2) == 5.0
    assert _avg_speed(10, 0) == 1000.0  # distance only — tiny epsilon minutes


def test_path_length_scales_with_frame():
    series = [
        {"ts_ms": 0, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "frame_w": 100, "frame_h": 0},
        {"ts_ms": 100, "x1": 100, "y1": 0, "x2": 100, "y2": 0, "frame_w": 100, "frame_h": 0},
    ]
    # frame_h 0 falls back to 720 → diagonal sqrt(100^2+720^2)
    assert abs(_path_length(series) - 13.7568) < 0.01


def test_team_aggregate_sums_and_sorts():
    rows = [
        {
            "team_id": "home",
            "goal": 2,
            "shot": 5,
            "pass": 10,
            "assist": 1,
            "dribble": 3,
            "pass_received": 8,
            "touches": 12,
            "possession_pct": 60.0,
        },
        {
            "team_id": "home",
            "goal": 0,
            "shot": 1,
            "pass": 4,
            "assist": 1,
            "dribble": 0,
            "pass_received": 3,
            "touches": 5,
            "possession_pct": 10.0,
        },
        {
            "team_id": "away",
            "goal": 1,
            "shot": 3,
            "pass": 6,
            "assist": 0,
            "dribble": 2,
            "pass_received": 5,
            "touches": 7,
            "possession_pct": 30.0,
        },
    ]
    teams = _aggregate_teams(rows)
    assert teams[0]["team_id"] == "home"
    assert teams[0]["goal"] == 2
    assert teams[0]["pass"] == 14
    assert teams[0]["players"] == 2
    assert teams[0]["possession_pct"] == 70.0
    assert teams[1]["team_id"] == "away"
    assert teams[1]["goal"] == 1


def test_stats_endpoint_caches_until_something_changes(client, register, monkeypatch):
    from uuid import UUID

    from app.core.database import get_db
    from app.main import app
    from app.models import Player, PlayerTrack, ProcessingJob, ProcessingRun, TrackIdentityLink
    from app.models.entities import CameraLabel, IdentityState, JobStage
    from app.services import match_stats

    owner = register("cache@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "M"}).json()
    base = f"/api/v1/facilities/{fac['id']}/matches/{match['id']}"
    db = next(app.dependency_overrides[get_db]())
    player = Player(facility_id=UUID(fac["id"]), display_name="P1")
    job = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="cache-test")
    db.add_all([player, job])
    db.flush()
    run = ProcessingRun(job_id=job.id, stage=JobStage.detect_track)
    db.add(run)
    db.flush()
    series = [{"ts_ms": t, "x1": 100, "y1": 100, "x2": 140, "y2": 200, "frame_w": 1920, "frame_h": 1080}
              for t in range(0, 2001, 100)]
    track = PlayerTrack(match_id=UUID(match["id"]), processing_run_id=run.id, camera_label=CameraLabel.A,
                        track_key="A-p1", start_ms=0, end_ms=2000, confidence=0.9,
                        features={"class": "person", "bbox_series": series})
    db.add(track)
    db.flush()
    db.add(TrackIdentityLink(track_id=track.id, player_id=player.id, state=IdentityState.proposed, confidence=0.9))
    db.commit()

    calls = []
    real = match_stats.build_match_stats
    monkeypatch.setattr(match_stats, "build_match_stats", lambda d, m: calls.append(m) or real(d, m))
    first = client.get(f"{base}/stats", headers=owner).json()
    assert client.get(f"{base}/stats", headers=owner).json() == first
    assert len(calls) == 1  # second request served from the cache

    client.patch(f"/api/v1/facilities/{fac['id']}/players/{player.id}", headers=owner, json={"display_name": "Ali"})
    renamed = client.get(f"{base}/stats", headers=owner).json()
    assert len(calls) == 2 and renamed["by_player"][0]["display_name"] == "Ali"
