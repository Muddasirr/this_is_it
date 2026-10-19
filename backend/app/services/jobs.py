from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import redis
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import Match, ProcessingJob
from app.models.entities import JobStatus, MatchStatus


QUEUE_NAME = "processing_jobs"


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def enqueue_processing_job(db: Session, match: Match, force: bool = False) -> ProcessingJob:
    existing = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.match_id == match.id,
            ProcessingJob.status.in_([JobStatus.queued, JobStatus.running]),
        )
        .first()
    )
    if existing and not force:
        return existing

    job = ProcessingJob(
        match_id=match.id,
        status=JobStatus.queued,
        progress_pct=0,
        idempotency_key=f"{match.id}:{uuid.uuid4()}",
    )
    db.add(job)
    match.status = MatchStatus.processing
    db.commit()
    db.refresh(job)

    client = get_redis()
    client.rpush(QUEUE_NAME, json.dumps({"job_id": str(job.id), "match_id": str(match.id)}))
    return job


def mark_job_running(db: Session, job: ProcessingJob) -> None:
    job.status = JobStatus.running
    job.started_at = datetime.now(UTC)
    db.commit()


def mark_job_progress(db: Session, job: ProcessingJob, stage, progress_pct: int) -> None:
    job.stage = stage
    job.progress_pct = progress_pct
    db.commit()


def mark_job_succeeded(db: Session, job: ProcessingJob, waiting_review: bool = True) -> None:
    job.status = JobStatus.waiting_review if waiting_review else JobStatus.succeeded
    job.progress_pct = 100
    job.finished_at = datetime.now(UTC)
    match = db.get(Match, job.match_id)
    if match:
        match.status = MatchStatus.needs_review if waiting_review else MatchStatus.published
    db.commit()


def mark_job_failed(db: Session, job: ProcessingJob, error: str) -> None:
    job.status = JobStatus.failed
    job.error_message = error
    job.finished_at = datetime.now(UTC)
    match = db.get(Match, job.match_id)
    if match:
        match.status = MatchStatus.failed
    db.commit()


def cancel_job(db: Session, job: ProcessingJob) -> ProcessingJob:
    if job.status not in {JobStatus.queued, JobStatus.running, JobStatus.waiting_review}:
        return job
    job.status = JobStatus.cancelled
    job.finished_at = datetime.now(UTC)
    job.error_message = "Cancelled by user"
    match = db.get(Match, job.match_id)
    if match and match.status == MatchStatus.processing:
        match.status = MatchStatus.ready_to_process
    db.commit()
    db.refresh(job)
    return job


def retry_job(db: Session, match: Match) -> ProcessingJob:
    return enqueue_processing_job(db, match, force=True)
