"""Match processing stages matching PRD §3 Processing workflow."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import (
    CameraSource,
    CameraSync,
    EventObservation,
    EventParticipant,
    HighlightClip,
    Match,
    MatchEvent,
    ModelVersion,
    PlayerTrack,
    ProcessingJob,
    ProcessingRun,
    TrackIdentityLink,
    UsageRecord,
    VideoAsset,
)
from app.models.entities import (
    CameraLabel,
    ConfirmationState,
    EventType,
    JobStage,
    JobStatus,
    VideoKind,
)
from app.core.config import get_settings
from app.services import jobs as job_service
from app.services.clips import render_and_store_clip
from app.services.avatars import attach_player_avatars
from app.services.identity import propose_identity_links
from app.services.media import estimate_offset_ms_stub, probe_video, validate_probe
from app.services.storage import download_to_path, object_key_for_proxy, upload_file
from app.services.vision import get_vision_backend
import secrets

logger = logging.getLogger(__name__)


STAGE_ORDER = [
    JobStage.ingest,
    JobStage.normalize,
    JobStage.synchronize,
    JobStage.detect_track,
    JobStage.resolve_identity,
    JobStage.infer_events,
    JobStage.fuse,
    JobStage.review_gate,
    JobStage.publish,
]


def _ensure_model_version(db: Session, backend) -> ModelVersion:
    existing = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == backend.name, ModelVersion.version == backend.version)
        .first()
    )
    if existing:
        return existing
    mv = ModelVersion(name=backend.name, version=backend.version, purpose="pipeline", config={})
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def _start_run(db: Session, job: ProcessingJob, stage: JobStage, model_version_id: UUID | None) -> ProcessingRun:
    run = ProcessingRun(
        job_id=job.id,
        stage=stage,
        status=JobStatus.running,
        model_version_id=model_version_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _finish_run(db: Session, run: ProcessingRun, status: JobStatus, checkpoint: dict | None = None) -> None:
    from datetime import UTC, datetime

    run.status = status
    run.checkpoint = checkpoint
    run.finished_at = datetime.now(UTC)
    db.commit()


def _clear_auto_analysis(db: Session, match_id: UUID) -> None:
    """Remove prior estimated analysis so reprocessing does not stack duplicates.

    Staff-confirmed events are kept. Tracks and unconfirmed events are replaced.
    """
    events = db.query(MatchEvent).filter(MatchEvent.match_id == match_id).all()
    for event in events:
        if event.confirmation == ConfirmationState.confirmed:
            continue
        db.query(EventParticipant).filter(EventParticipant.event_id == event.id).delete(synchronize_session=False)
        db.query(EventObservation).filter(EventObservation.event_id == event.id).delete(synchronize_session=False)
        db.query(HighlightClip).filter(HighlightClip.event_id == event.id).delete(synchronize_session=False)
        db.delete(event)

    tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == match_id).all()
    track_ids = [t.id for t in tracks]
    if track_ids:
        db.query(TrackIdentityLink).filter(TrackIdentityLink.track_id.in_(track_ids)).delete(
            synchronize_session=False
        )
        for t in tracks:
            db.delete(t)
    db.commit()


def _participant_roles(
    event_type: str, track_keys: list[str], evidence: dict | None
) -> list[tuple[str, str]]:
    evidence = evidence or {}
    if event_type == "pass":
        passer = evidence.get("passer") or (track_keys[0] if track_keys else None)
        receiver = evidence.get("receiver") or (track_keys[1] if len(track_keys) > 1 else None)
        roles: list[tuple[str, str]] = []
        if passer:
            roles.append((passer, "passer"))
        if receiver and receiver != passer:
            roles.append((receiver, "receiver"))
        return roles
    if event_type == "shot":
        shooter = evidence.get("shooter") or (track_keys[0] if track_keys else None)
        return [(shooter, "shooter")] if shooter else []
    if event_type == "goal":
        scorer = evidence.get("shooter") or (track_keys[0] if track_keys else None)
        return [(scorer, "scorer")] if scorer else []
    return [(k, "primary" if i == 0 else "secondary") for i, k in enumerate(track_keys)]


def process_job(db: Session, job_id: UUID) -> None:
    job = db.get(ProcessingJob, job_id)
    if not job:
        logger.error("Job %s not found", job_id)
        return
    if job.status not in {JobStatus.queued, JobStatus.running}:
        logger.info("Skipping job %s in status %s", job_id, job.status)
        return

    job_service.mark_job_running(db, job)
    backend = get_vision_backend()
    model = _ensure_model_version(db, backend)

    try:
        settings = get_settings()
        originals = (
            db.query(VideoAsset)
            .filter(VideoAsset.match_id == job.match_id, VideoAsset.kind == VideoKind.original)
            .all()
        )
        if not originals:
            raise RuntimeError("Need at least one original camera upload")

        # Map camera label -> asset
        labeled: dict[CameraLabel, VideoAsset] = {}
        for asset in originals:
            cam = db.get(CameraSource, asset.camera_source_id) if asset.camera_source_id else None
            if cam:
                labeled[cam.label] = asset
        if not labeled:
            raise RuntimeError("No labeled camera assets found")
        if (
            not settings.allow_single_camera
            and (CameraLabel.A not in labeled or CameraLabel.B not in labeled)
        ):
            raise RuntimeError("Camera A and B assets required")

        with tempfile.TemporaryDirectory(prefix="futsal-") as tmp:
            tmp_path = Path(tmp)
            local: dict[CameraLabel, Path] = {}
            probes: dict[CameraLabel, dict] = {}

            # 1. Ingest / validate
            run = _start_run(db, job, JobStage.ingest, model.id)
            job_service.mark_job_progress(db, job, JobStage.ingest, 10)
            for label, asset in labeled.items():
                path = tmp_path / f"{label.value}.mp4"
                download_to_path(asset.object_key, path)
                local[label] = path
                try:
                    probe = probe_video(path)
                    errors = validate_probe(probe)
                except Exception as exc:  # noqa: BLE001
                    asset.validation_status = "failed"
                    asset.validation_errors = {"error": str(exc)}
                    db.commit()
                    raise
                asset.duration_ms = probe["duration_ms"]
                asset.width = probe["width"]
                asset.height = probe["height"]
                asset.frame_rate = probe["frame_rate"]
                asset.codec = probe["codec"]
                asset.has_audio = probe["has_audio"]
                asset.validation_status = "failed" if errors else "passed"
                asset.validation_errors = {"errors": errors} if errors else None
                probes[label] = probe
                db.commit()
                if errors:
                    raise RuntimeError(f"Camera {label.value} validation failed: {errors}")
            _finish_run(db, run, JobStatus.succeeded, {"probes": {k.value: v for k, v in probes.items()}})

            # 2. Normalize proxies (best-effort if ffmpeg available)
            run = _start_run(db, job, JobStage.normalize, model.id)
            job_service.mark_job_progress(db, job, JobStage.normalize, 25)
            from app.models import Match

            match = db.get(Match, job.match_id)
            for label, path in local.items():
                proxy_path = tmp_path / f"{label.value}_proxy.mp4"
                try:
                    from app.services.media import make_proxy

                    make_proxy(path, proxy_path)
                    key = object_key_for_proxy(match.facility_id, match.id, label.value)
                    upload_file(proxy_path, key)
                    db.add(
                        VideoAsset(
                            match_id=match.id,
                            camera_source_id=labeled[label].camera_source_id,
                            kind=VideoKind.proxy,
                            object_key=key,
                            content_type="video/mp4",
                            validation_status="passed",
                        )
                    )
                    db.commit()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Proxy generation skipped for %s: %s", label, exc)
            _finish_run(db, run, JobStatus.succeeded)

            # 3. Synchronize
            run = _start_run(db, job, JobStage.synchronize, model.id)
            job_service.mark_job_progress(db, job, JobStage.synchronize, 35)
            if CameraLabel.A in probes and CameraLabel.B in probes:
                offset_ms, confidence, method = estimate_offset_ms_stub(
                    probes[CameraLabel.A]["duration_ms"],
                    probes[CameraLabel.B]["duration_ms"],
                )
                notes = "Auto estimate — staff should verify in review UI"
            else:
                offset_ms, confidence, method = 0, 0.0, "single_camera"
                notes = "Single-camera Phase 0 run — sync skipped"
            db.add(
                CameraSync(
                    match_id=job.match_id,
                    offset_ms=offset_ms,
                    confidence=confidence,
                    method=method,
                    is_manual=False,
                    notes=notes,
                )
            )
            db.commit()
            _finish_run(db, run, JobStatus.succeeded, {"offset_ms": offset_ms, "confidence": confidence})

            # 4. Detect / track (replace prior auto tracks/events for this match)
            run = _start_run(db, job, JobStage.detect_track, model.id)
            job_service.mark_job_progress(db, job, JobStage.detect_track, 55)
            _clear_auto_analysis(db, job.match_id)
            all_detections = []
            vision_stats = []
            for label, path in local.items():
                dets = backend.detect_and_track(str(path), label.value)
                vision_stats.append(getattr(backend, "last_run_stats", {}))
                for d in dets:
                    feats = dict(d.features or {})
                    feats["class"] = d.class_name
                    feats["bbox_series"] = d.bbox_series
                    db.add(
                        PlayerTrack(
                            match_id=job.match_id,
                            processing_run_id=run.id,
                            camera_label=label,
                            track_key=d.track_key,
                            start_ms=d.start_ms,
                            end_ms=d.end_ms,
                            confidence=d.confidence,
                            features=feats,
                        )
                    )
                all_detections.extend(dets)
            db.commit()
            _finish_run(
                db,
                run,
                JobStatus.succeeded,
                {"track_count": len(all_detections), "vision_stats": vision_stats},
            )

            # 5. Identity proposals for staff correction
            run = _start_run(db, job, JobStage.resolve_identity, model.id)
            job_service.mark_job_progress(db, job, JobStage.resolve_identity, 65)
            saved_tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == job.match_id).all()
            # clear prior auto proposals for this match's tracks
            track_ids = [t.id for t in saved_tracks]
            if track_ids:
                db.query(TrackIdentityLink).filter(TrackIdentityLink.track_id.in_(track_ids)).delete(
                    synchronize_session=False
                )
                db.commit()
            links = propose_identity_links(db, job.match_id, saved_tracks)
            match_row = db.get(Match, job.match_id)
            avatar_count = 0
            if match_row:
                avatar_count = attach_player_avatars(db, match_row, saved_tracks)
            track_by_key = {t.track_key: t for t in saved_tracks}
            _finish_run(
                db,
                run,
                JobStatus.succeeded,
                {"identity_links": len(links), "avatars": avatar_count},
            )

            # 6. Infer events
            run = _start_run(db, job, JobStage.infer_events, model.id)
            job_service.mark_job_progress(db, job, JobStage.infer_events, 80)
            candidates = []
            for label in local:
                label_dets = [d for d in all_detections if d.camera_label == label.value]
                candidates.extend(backend.infer_events(label_dets, label.value))

            # 7. Fuse across cameras A/B/C
            run_fuse = _start_run(db, job, JobStage.fuse, model.id)
            job_service.mark_job_progress(db, job, JobStage.fuse, 90)
            from app.services.fuse import fuse_event_candidates

            # Cross-cam A/B/C: keep best obs per type+time+player bucket
            fused_list = fuse_event_candidates(candidates, bucket_ms=1500)
            fused: dict[tuple, list] = {}
            for c in fused_list:
                key = (c.event_type, c.camera_ts_ms // 1000)
                fused.setdefault(key, []).append(c)

            type_map = {
                "goal": EventType.goal,
                "shot": EventType.shot,
                "pass": EventType.pass_event,
            }
            created_events: list[MatchEvent] = []
            for (_etype, _sec), obs_list in fused.items():
                primary = max(obs_list, key=lambda x: x.confidence)
                event = MatchEvent(
                    match_id=job.match_id,
                    processing_run_id=run.id,
                    event_type=type_map.get(primary.event_type, EventType.shot),
                    canonical_ts_ms=primary.camera_ts_ms,
                    confirmation=ConfirmationState.estimated,
                    confidence=primary.confidence,
                    provenance={"model": backend.name, "version": backend.version, "heuristic": True},
                    is_canonical=True,
                )
                db.add(event)
                db.flush()
                created_events.append(event)
                for obs in obs_list:
                    db.add(
                        EventObservation(
                            event_id=event.id,
                            camera_label=CameraLabel(obs.camera_label),
                            camera_ts_ms=obs.camera_ts_ms,
                            confidence=obs.confidence,
                            payload=obs.evidence,
                        )
                    )
                roles = _participant_roles(primary.event_type, primary.track_keys, primary.evidence)
                for tkey, role in roles:
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
                            confidence=primary.confidence,
                        )
                    )
            db.commit()
            _finish_run(db, run, JobStatus.succeeded)
            _finish_run(db, run_fuse, JobStatus.succeeded, {"canonical_events": len(fused)})

            # 8. Review gate — auto evidence clips for uncertain events
            run = _start_run(db, job, JobStage.review_gate, model.id)
            job_service.mark_job_progress(db, job, JobStage.review_gate, 95)
            first_label = next(iter(local))
            src_path = local[first_label]
            for event in created_events[:25]:
                start_ms = max(0, event.canonical_ts_ms - 3000)
                end_ms = event.canonical_ts_ms + 4000
                key = f"facilities/{match.facility_id}/matches/{match.id}/clips/{event.id}.mp4"
                try:
                    render_and_store_clip(src_path, key, start_ms, end_ms)
                    clip_asset = VideoAsset(
                        match_id=match.id,
                        camera_source_id=labeled[first_label].camera_source_id,
                        kind=VideoKind.clip,
                        object_key=key,
                        content_type="video/mp4",
                        validation_status="passed",
                    )
                    db.add(clip_asset)
                    db.flush()
                    db.add(
                        HighlightClip(
                            match_id=match.id,
                            event_id=event.id,
                            video_asset_id=clip_asset.id,
                            title=f"{'pass' if event.event_type.name == 'pass_event' else event.event_type.value} @ {event.canonical_ts_ms}ms",
                            start_ms=start_ms,
                            end_ms=end_ms,
                            share_token=secrets.token_urlsafe(12),
                        )
                    )
                    from app.models import EventObservation as EO

                    obs = db.query(EO).filter(EO.event_id == event.id).first()
                    if obs:
                        obs.evidence_key = key
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Clip render failed for %s: %s", event.id, exc)
            db.commit()
            _finish_run(db, run, JobStatus.succeeded, {"clips_attempted": min(25, len(created_events))})

            run = _start_run(db, job, JobStage.publish, model.id)
            _finish_run(db, run, JobStatus.succeeded, {"note": "awaiting staff review before published stats"})

            db.add(
                UsageRecord(
                    facility_id=match.facility_id,
                    match_id=match.id,
                    category="gpu_seconds",
                    quantity=0,
                    unit_cost_cents=0,
                    details={"note": "stub — fill from real GPU timing"},
                )
            )
            db.commit()

        job_service.mark_job_succeeded(db, job, waiting_review=True)
        logger.info("Job %s completed — awaiting review", job_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        job_service.mark_job_failed(db, job, str(exc))
