"""Inject precomputed detections (Kaggle/offline) into a match — skip YOLO.

Run inside the api container:
  python -m app.scripts.inject_detections <match_id> /path/to/detections.json [/path/to/events.json]
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.api.share import ensure_match_share
from app.core.database import SessionLocal
from app.models import (
    EventObservation,
    EventParticipant,
    Match,
    MatchEvent,
    ModelVersion,
    PlayerTrack,
    ProcessingJob,
    ProcessingRun,
    TrackIdentityLink,
)
from app.models.entities import (
    CameraLabel,
    ConfirmationState,
    EventType,
    JobStage,
    JobStatus,
    MatchStatus,
)
from app.services.avatars import attach_player_avatars
from app.services.events import infer_event_candidates
from app.services.fuse import fuse_event_candidates
from app.services.identity import propose_identity_links
from app.services.vision import Detection, EventCandidate
from app.workers.pipeline import _clear_auto_analysis, _participant_roles


def _ensure_model(db: Session) -> ModelVersion:
    existing = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == "courtlens-offline", ModelVersion.version == "kaggle")
        .first()
    )
    if existing:
        return existing
    mv = ModelVersion(name="courtlens-offline", version="kaggle", purpose="pipeline", config={})
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def _load_detections(path: Path) -> list[Detection]:
    raw = json.loads(path.read_text())
    out: list[Detection] = []
    for d in raw:
        feats = dict(d.get("features") or {})
        feats.setdefault("class", d.get("class_name", "person"))
        if d.get("bbox_series") is not None:
            feats["bbox_series"] = d["bbox_series"]
        out.append(
            Detection(
                track_key=d["track_key"],
                camera_label=d.get("camera_label") or "A",
                start_ms=int(d["start_ms"]),
                end_ms=int(d["end_ms"]),
                confidence=float(d.get("confidence") or 0.5),
                class_name=d.get("class_name") or "person",
                features=feats,
                bbox_series=d.get("bbox_series") or feats.get("bbox_series") or [],
            )
        )
    return out


def inject_match_from_files(
    db: Session,
    match_id: UUID,
    detections_path: Path,
    events_path: Path | None = None,
    *,
    publish: bool = True,
) -> dict:
    match = db.get(Match, match_id)
    if not match:
        raise RuntimeError(f"match {match_id} not found")

    detections = _load_detections(detections_path)
    model = _ensure_model(db)
    job = ProcessingJob(
        match_id=match_id,
        status=JobStatus.running,
        stage=JobStage.detect_track,
        progress_pct=50,
        idempotency_key=f"inject-{match_id}-{secrets.token_hex(8)}",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run = ProcessingRun(job_id=job.id, stage=JobStage.detect_track, status=JobStatus.running, model_version_id=model.id)
    db.add(run)
    db.commit()
    db.refresh(run)

    _clear_auto_analysis(db, match_id)

    for d in detections:
        feats = dict(d.features or {})
        feats["class"] = d.class_name
        feats["bbox_series"] = d.bbox_series
        db.add(
            PlayerTrack(
                match_id=match_id,
                processing_run_id=run.id,
                camera_label=CameraLabel(d.camera_label if d.camera_label in {"A", "B", "C"} else "A"),
                track_key=d.track_key,
                start_ms=d.start_ms,
                end_ms=d.end_ms,
                confidence=d.confidence,
                features=feats,
            )
        )
    db.commit()
    run.status = JobStatus.succeeded
    db.commit()

    saved_tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == match_id).all()
    propose_identity_links(db, match_id, saved_tracks)
    attach_player_avatars(db, match, saved_tracks)
    track_by_key = {t.track_key: t for t in saved_tracks}

    if events_path and events_path.exists():
        raw_ev = json.loads(events_path.read_text())
        candidates = [
            EventCandidate(
                event_type=e["event_type"],
                camera_label=e.get("camera_label") or "A",
                camera_ts_ms=int(e["camera_ts_ms"]),
                confidence=float(e.get("confidence") or 0.5),
                track_keys=list(e.get("track_keys") or []),
                evidence=e.get("evidence") or {},
            )
            for e in raw_ev
        ]
    else:
        candidates = infer_event_candidates(detections, "A")

    fused_list = fuse_event_candidates(candidates, bucket_ms=1500)
    type_map = {
        "goal": EventType.goal,
        "shot": EventType.shot,
        "pass": EventType.pass_event,
    }
    evt_run = ProcessingRun(
        job_id=job.id, stage=JobStage.infer_events, status=JobStatus.running, model_version_id=model.id
    )
    db.add(evt_run)
    db.commit()
    db.refresh(evt_run)

    for c in fused_list:
        event = MatchEvent(
            match_id=match_id,
            processing_run_id=evt_run.id,
            event_type=type_map.get(c.event_type, EventType.shot),
            canonical_ts_ms=c.camera_ts_ms,
            confirmation=ConfirmationState.estimated,
            confidence=c.confidence,
            provenance={"model": "courtlens-offline", "version": "kaggle", "injected": True},
            is_canonical=True,
        )
        db.add(event)
        db.flush()
        db.add(
            EventObservation(
                event_id=event.id,
                camera_label=CameraLabel.A,
                camera_ts_ms=c.camera_ts_ms,
                confidence=c.confidence,
                payload=c.evidence,
            )
        )
        for tkey, role in _participant_roles(c.event_type, c.track_keys, c.evidence):
            track = track_by_key.get(tkey)
            player_id = None
            if track:
                link = (
                    db.query(TrackIdentityLink)
                    .filter(TrackIdentityLink.track_id == track.id)
                    .order_by(TrackIdentityLink.created_at.desc())
                    .first()
                )
                if link:
                    player_id = link.player_id
            db.add(
                EventParticipant(
                    event_id=event.id,
                    player_id=player_id,
                    track_id=track.id if track else None,
                    role=role,
                    confidence=c.confidence,
                )
            )
    evt_run.status = JobStatus.succeeded
    job.status = JobStatus.succeeded
    job.stage = JobStage.publish
    job.progress_pct = 100
    match.status = MatchStatus.needs_review
    db.commit()

    share_token = None
    if publish:
        match.status = MatchStatus.published
        db.commit()
        share = ensure_match_share(db, match_id)
        share_token = share.share_token

    return {
        "match_id": str(match_id),
        "tracks": len(saved_tracks),
        "events": len(fused_list),
        "share_token": share_token,
        "report_path": f"/report/{share_token}" if share_token else None,
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m app.scripts.inject_detections <match_id> <detections.json> [events.json]")
    match_id = UUID(sys.argv[1])
    dets = Path(sys.argv[2])
    ev = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    db = SessionLocal()
    try:
        result = inject_match_from_files(db, match_id, dets, ev, publish=True)
        print(json.dumps(result, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
