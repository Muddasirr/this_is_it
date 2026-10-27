"""Redis queue consumer for processing jobs."""

from __future__ import annotations

import json
import logging
import signal
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.core.database import SessionLocal
from app.services.jobs import QUEUE_NAME, get_redis
from app.workers.pipeline import process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("worker")

_running = True
ORPHAN_MAX_AGE = timedelta(hours=12)  # a Kaggle run is capped at 6 h; older ⇒ abandoned


def _stop(*_: object) -> None:
    global _running
    _running = False
    logger.info("Shutting down worker…")


def requeue_orphans(client) -> list[str]:
    """Jobs left 'running' by a killed worker (one worker ⇒ nothing else is running them): put recent
    ones back on the queue — a Kaggle job re-attaches to its GPU run (kaggle_backend) — and fail the
    stale ones rather than silently re-processing (and re-billing) old matches."""
    from app.models import ProcessingJob
    from app.models.entities import JobStatus
    from app.services.jobs import mark_job_failed

    db = SessionLocal()
    requeued = []
    try:
        cutoff = datetime.now(UTC) - ORPHAN_MAX_AGE
        for job in db.query(ProcessingJob).filter(ProcessingJob.status == JobStatus.running).all():
            started = job.started_at or job.created_at
            if started and started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            if started and started < cutoff:
                logger.info("Failing job %s orphaned since %s", job.id, started)
                mark_job_failed(db, job, "Worker restarted; job was orphaned too long ago — retry the match")
                continue
            logger.info("Re-queueing job %s orphaned by a worker restart", job.id)
            client.rpush(QUEUE_NAME, json.dumps({"job_id": str(job.id)}))
            requeued.append(str(job.id))
    finally:
        db.close()
    return requeued


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    client = get_redis()
    requeue_orphans(client)
    logger.info("Worker listening on queue %s", QUEUE_NAME)

    while _running:
        item = client.blpop(QUEUE_NAME, timeout=2)
        if not item:
            continue
        _, payload = item
        try:
            data = json.loads(payload)
            job_id = UUID(data["job_id"])
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.error("Bad queue payload: %s", payload)
            continue

        db = SessionLocal()
        try:
            process_job(db, job_id)
        finally:
            db.close()

    # brief pause so logs flush cleanly
    time.sleep(0.1)


if __name__ == "__main__":
    main()
