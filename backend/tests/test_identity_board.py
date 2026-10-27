import uuid


def test_merge_player_and_mark_not_a_player(client, register):
    from app.core.database import get_db
    from app.main import app
    from app.models import PlayerTrack
    from app.models.entities import CameraLabel

    owner = register("board@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Board"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "Sun"}).json()
    p1 = client.post(f"/api/v1/facilities/{fac['id']}/players", headers=owner, json={"display_name": "Ana"}).json()
    db = next(app.dependency_overrides[get_db]())
    tracks = [PlayerTrack(match_id=uuid.UUID(match["id"]), processing_run_id=uuid.uuid4(), camera_label=CameraLabel.A,
                          track_key=f"A-{i}", start_ms=i * 1000, end_ms=i * 1000 + 900,
                          features={"class": "person", "thumb_key": f"thumbs/x/{i}.jpg"}) for i in range(3)]
    db.add_all(tracks)
    db.commit()
    ids = [str(t.id) for t in tracks]
    base = f"/api/v1/facilities/{fac['id']}/matches/{match['id']}"

    r = client.put(f"{base}/tracks/identity", headers=owner, json={"track_ids": ids[:2], "player_id": p1["id"]})
    assert r.status_code == 200 and r.json()["updated"] == 2
    r = client.put(f"{base}/tracks/identity", headers=owner, json={"track_ids": [ids[2]], "player_id": None})
    assert r.status_code == 200
    listed = {t["id"]: t for t in client.get(f"{base}/tracks", headers=owner).json()}
    assert [listed[i]["identity"]["player_id"] for i in ids] == [p1["id"], p1["id"], None]
    assert listed[ids[2]]["identity"]["state"] == "unknown"
    assert "thumb_url" in listed[ids[0]]
    # A track of another match is refused
    r = client.put(f"{base}/tracks/identity", headers=owner, json={"track_ids": [str(uuid.uuid4())], "player_id": None})
    assert r.status_code == 404
