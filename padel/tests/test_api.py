"""Upload -> calibrate -> analyse -> report -> review, with detections injected (no YOLO on CPU)."""
import importlib
import time

import cv2
import numpy as np
import pytest
from conftest import synthetic_detections

from padel_analyzer.engine import court as C


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PADEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PADEL_DB_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("PADEL_BACKEND", "local")  # detections are injected; no Kaggle round trip
    import padel_analyzer.api.settings as settings

    importlib.reload(settings)
    import padel_analyzer.api.db as db

    importlib.reload(db)
    import padel_analyzer.api.worker as worker

    importlib.reload(worker)
    import padel_analyzer.api.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app), settings


def _video(path, n=90):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    img = np.full((1080, 1920, 3), (180, 90, 30), np.uint8)
    for _ in range(n):
        w.write(img)
    w.release()


def test_end_to_end(client, tmp_path, calib):
    c, settings = client
    src = tmp_path / "clip.mp4"
    _video(src)
    r = c.post("/api/matches", files={"file": ("clip.mp4", src.read_bytes(), "video/mp4")}, data={"title": "Demo"})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["n_frames"] == 90 and m["width"] == 1920 and m["status"] == "uploaded"

    assert c.post(f"/api/matches/{m['id']}/analyze").status_code == 409  # not calibrated

    pts = dict(zip(C.LANDMARKS, calib.to_px(np.array(list(C.LANDMARKS.values()))).tolist()))
    four = {k: pts[k] for k in ["far_service_left", "far_service_right", "near_service_left", "near_service_right"]}
    r = c.put(f"/api/matches/{m['id']}/calibration", json={"points": four, "save_camera": "Court 1"})
    assert r.status_code == 200, r.text
    assert r.json()["match"]["status"] == "calibrated"
    assert len(c.get("/api/cameras").json()) == 1
    assert c.get(f"/api/matches/{m['id']}/frame", params={"t": 1.0, "overlay": True}).headers["content-type"] == "image/jpeg"

    det, _ = synthetic_detections(calib, n=90)
    det.save(settings.MATCH_DIR / m["id"] / "detections.npz")
    assert c.post(f"/api/matches/{m['id']}/analyze").status_code == 200
    for _ in range(100):
        st = c.get(f"/api/matches/{m['id']}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "done", st.get("error")

    rep = c.get(f"/api/matches/{m['id']}/report").json()
    assert len(rep["players"]) == 4
    r = c.patch(f"/api/matches/{m['id']}/review", json={"players": {"0": {"name": "Ari"}}, "rallies": {"0": {"winner_team": 1}}})
    assert r.status_code == 200
    rep = c.get(f"/api/matches/{m['id']}/report").json()
    assert rep["players"][0]["name"] == "Ari"
    if rep["rallies"]:
        assert rep["teams"][1]["stats"]["rallies_won"] == 1

    v = c.get(f"/api/matches/{m['id']}/video", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100


def test_exports_and_token(client, tmp_path, calib, monkeypatch):
    c, settings = client
    src = tmp_path / "clip.mp4"
    _video(src, n=60)
    m = c.post("/api/matches", files={"file": ("clip.mp4", src.read_bytes(), "video/mp4")}).json()
    pts = dict(zip(C.LANDMARKS, calib.to_px(np.array(list(C.LANDMARKS.values()))).tolist()))
    c.put(f"/api/matches/{m['id']}/calibration", json={"points": pts})
    det, _ = synthetic_detections(calib, n=60)
    det.save(settings.MATCH_DIR / m["id"] / "detections.npz")
    c.post(f"/api/matches/{m['id']}/analyze")
    for _ in range(100):
        if c.get(f"/api/matches/{m['id']}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    for table in ("shots", "players", "pairs", "rallies"):
        r = c.get(f"/api/matches/{m['id']}/export/{table}.csv")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    players = c.get(f"/api/matches/{m['id']}/export/players.csv").text.splitlines()
    assert len(players) == 5 and "distance_m" in players[0] and "net_approaches" in players[0]
    assert c.get(f"/api/matches/{m['id']}/export/nope.csv").status_code == 404

    import padel_analyzer.api.main as main

    monkeypatch.setattr(main, "API_TOKEN", "s3cret")
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/matches").status_code == 401
    assert c.get("/api/matches", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get(f"/api/matches/{m['id']}/video", params={"token": "s3cret"}).status_code == 200
