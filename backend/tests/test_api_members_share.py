"""API: facility staff management + revocable share links (SQLite, no lifespan)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def client():
    from app.core.database import Base, get_db
    from app.main import app

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def _db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db
    yield TestClient(app)  # no `with`: skip lifespan (Postgres patches / bucket)
    app.dependency_overrides.clear()


def _user(client, email: str) -> dict:
    client.post("/api/v1/auth/register", json={"email": email, "full_name": email, "password": "Testpass123!"})
    token = client.post("/api/v1/auth/login", json={"email": email, "password": "Testpass123!"}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}


def test_owner_adds_staff_and_staff_cannot_add(client):
    owner = _user(client, "owner@example.com")
    staff = _user(client, "staff@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court 1"}).json()
    base = f"/api/v1/facilities/{fac['id']}"

    assert client.get(base, headers=staff).status_code == 403
    r = client.post(f"{base}/members", headers=owner, json={"email": "STAFF@EXAMPLE.com", "role": "staff"})
    assert r.status_code == 201 and r.json()["role"] == "staff"
    assert client.get(base, headers=staff).status_code == 200
    emails = {m["email"]: m["role"] for m in client.get(f"{base}/members", headers=staff).json()}
    assert emails == {"owner@example.com": "owner", "staff@example.com": "staff"}

    _user(client, "third@example.com")
    r = client.post(f"{base}/members", headers=staff, json={"email": "third@example.com"})
    assert r.status_code == 403
    r = client.post(f"{base}/members", headers=owner, json={"email": "nobody@example.com"})
    assert r.status_code == 404


def test_revoked_share_link_stops_working_and_publish_mints_new_one(client):
    owner = _user(client, "owner2@example.com")
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
