"""Redis queue consumer for processing jobs."""

from __future__ import annotations

import json
import logging
import signal
import time
from uuid import UUID

from app.core.database import SessionLocal
from app.services.jobs import QUEUE_NAME, get_redis
from app.workers.pipeline import process_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("worker")

_running = True


def _stop(*_: object) -> None:
    global _running
    _running = False
    logger.info("Shutting down worker…")


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    client = get_redis()
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
