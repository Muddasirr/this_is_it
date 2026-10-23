"""Shared API fixtures: FastAPI app on in-memory SQLite (no lifespan / Postgres / Redis)."""

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


@pytest.fixture
def register(client):
    """register("x@example.com") → auth headers for a new user."""

    def _register(email: str) -> dict:
        return _register_user(client, email)

    return _register


def _register_user(client, email: str) -> dict:
    client.post("/api/v1/auth/register", json={"email": email, "full_name": email, "password": "Testpass123!"})
    token = client.post("/api/v1/auth/login", json={"email": email, "password": "Testpass123!"}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}
