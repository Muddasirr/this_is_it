"""API: facility staff management + revocable share links (SQLite, no lifespan)."""

def test_owner_adds_staff_and_staff_cannot_add(client, register):
    owner = register("owner@example.com")
    staff = register("staff@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court 1"}).json()
    base = f"/api/v1/facilities/{fac['id']}"

    assert client.get(base, headers=staff).status_code == 403
    r = client.post(f"{base}/members", headers=owner, json={"email": "STAFF@EXAMPLE.com", "role": "staff"})
    assert r.status_code == 201 and r.json()["role"] == "staff"
    assert client.get(base, headers=staff).status_code == 200
    emails = {m["email"]: m["role"] for m in client.get(f"{base}/members", headers=staff).json()}
    assert emails == {"owner@example.com": "owner", "staff@example.com": "staff"}

    register("third@example.com")
    r = client.post(f"{base}/members", headers=staff, json={"email": "third@example.com"})
    assert r.status_code == 403
    r = client.post(f"{base}/members", headers=owner, json={"email": "nobody@example.com"})
    assert r.status_code == 404


def test_revoked_share_link_stops_working_and_publish_mints_new_one(client, register):
    owner = register("owner2@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court 2"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "Fri"}).json()
    mbase = f"/api/v1/facilities/{fac['id']}/matches/{match['id']}"

    first = client.post(f"{mbase}/publish", headers=owner).json()["share_token"]
    assert client.get(f"/api/v1/share/matches/{first}").status_code == 200

    assert client.post(f"{mbase}/share/revoke", headers=owner).json()["revoked"] == 1
    assert client.get(f"/api/v1/share/matches/{first}").status_code == 404

    second = client.post(f"{mbase}/publish", headers=owner).json()["share_token"]
    assert second != first
    assert client.get(f"/api/v1/share/matches/{second}").status_code == 200


def test_run_stats_shown_for_match_awaiting_review(client, register):
    from uuid import UUID

    from app.core.database import get_db
    from app.main import app
    from app.models import ProcessingJob, ProcessingRun
    from app.models.entities import JobStage, JobStatus

    owner = register("runstats@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "C"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "M"}).json()
    db = next(app.dependency_overrides[get_db]())
    job = ProcessingJob(match_id=UUID(match["id"]), status=JobStatus.waiting_review, idempotency_key="rs")
    db.add(job)
    db.flush()
    db.add(ProcessingRun(job_id=job.id, stage=JobStage.detect_track, status=JobStatus.succeeded,
                         checkpoint={"vision_stats": [{"camera_label": "A", "person_tracks": 5, "imgsz": 1280}]}))
    db.commit()
    r = client.get(f"/api/v1/facilities/{fac['id']}/matches/{match['id']}/run-stats", headers=owner).json()
    assert r["available"] and r["cameras"][0]["person_tracks"] == 5
