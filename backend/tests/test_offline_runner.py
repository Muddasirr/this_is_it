"""End-to-end: the real process_job on SQLite via app.scripts.run_offline (stub vision)."""

import json

import pytest


@pytest.fixture
def stub_media(monkeypatch):
    from app.workers import pipeline

    probe = {
        "duration_ms": 30_000,
        "width": 1920,
        "height": 1080,
        "frame_rate": 25.0,
        "codec": "h264",
        "has_audio": False,
        "format_name": "mp4",
        "bit_rate": None,
    }
    monkeypatch.setattr(pipeline, "probe_video", lambda _p: dict(probe))
    monkeypatch.setattr(pipeline, "estimate_offset_ms", lambda *_a, **_k: (1200, 0.8, "motion_xcorr"))
    monkeypatch.setenv("VISION_BACKEND", "stub")
    yield
    from app.core.config import get_settings

    get_settings.cache_clear()


def test_run_offline_two_cameras(tmp_path, stub_media):
    from app.scripts.run_offline import main

    vids = []
    for label in "AB":
        p = tmp_path / f"cam_{label}.mp4"
        p.write_bytes(b"\0" * 64)
        vids.append(f"{label}={p}")
    out = tmp_path / "out"
    summary = main(["--out", str(out), "--camera", vids[0], "--camera", vids[1]])

    assert summary["job_status"] == "waiting_review"
    assert summary["sync"] == [{"camera": "B", "offset_ms": 1200, "confidence": 0.8, "method": "motion_xcorr"}]
    dets = json.loads((out / "detections.json").read_text())
    assert {d["camera_label"] for d in dets} == {"A", "B"}
    for name in ("events.json", "canonical_events.json", "players.json", "match_stats.json"):
        assert (out / name).is_file()


def test_replay_reruns_post_vision_stages(tmp_path, stub_media):
    from app.scripts.run_offline import main

    vids = []
    for label in "AB":
        p = tmp_path / f"cam_{label}.mp4"
        p.write_bytes(b"\0" * 64)
        vids += ["--camera", f"{label}={p}"]
    first = tmp_path / "first"
    main(["--out", str(first), *vids])

    again = main(["--replay", str(first), "--out", str(tmp_path / "again")])
    assert again["job_status"] == "waiting_review"
    assert again["sync"][0]["offset_ms"] == 1200  # carried over from the saved run
    assert again["counts"]["tracks"] == json.loads((first / "summary.json").read_text())["counts"]["tracks"]
