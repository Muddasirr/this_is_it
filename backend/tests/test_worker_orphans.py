from datetime import UTC, datetime, timedelta
from uuid import UUID


def test_restart_requeues_recent_orphans_and_fails_stale_ones(client, register, monkeypatch):
    """A worker killed mid-job leaves it 'running' forever: on startup recent ones go back on the
    queue (Kaggle re-attaches), old ones fail instead of silently re-billing GPU time."""
    from app.core.database import get_db
    from app.main import app
    from app.models import ProcessingJob
    from app.models.entities import JobStatus
    from app.workers import runner

    owner = register("orphan@example.com")
    fac = client.post("/api/v1/facilities", headers=owner, json={"name": "Court"}).json()
    match = client.post(f"/api/v1/facilities/{fac['id']}/matches", headers=owner, json={"title": "M"}).json()
    db = next(app.dependency_overrides[get_db]())
    now = datetime.now(UTC)
    recent = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="r", status=JobStatus.running,
                           started_at=now - timedelta(hours=1))
    stale = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="s", status=JobStatus.running,
                          started_at=now - timedelta(days=2))
    done = ProcessingJob(match_id=UUID(match["id"]), idempotency_key="d", status=JobStatus.waiting_review)
    db.add_all([recent, stale, done])
    db.commit()
    monkeypatch.setattr(runner, "SessionLocal", lambda: next(app.dependency_overrides[get_db]()))

    class Redis:
        pushed: list = []

        def rpush(self, queue, payload):
            self.pushed.append(payload)

    r = Redis()
    assert runner.requeue_orphans(r) == [str(recent.id)]
    assert len(r.pushed) == 1
    db.expire_all()
    assert db.get(ProcessingJob, stale.id).status == JobStatus.failed
    assert db.get(ProcessingJob, done.id).status == JobStatus.waiting_review
