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


def _box_score(pt: dict, conf_fallback: float = 0.5) -> float:
    try:
        w = float(pt.get("x2", 0)) - float(pt.get("x1", 0))
        h = float(pt.get("y2", 0)) - float(pt.get("y1", 0))
        return w * h * float(pt.get("conf") or conf_fallback)
    except (TypeError, ValueError):
        return -1.0


def _top_boxes(series: list[dict], conf_fallback: float = 0.5, k: int = 8) -> list[dict]:
    ranked = sorted(series, key=lambda pt: _box_score(pt, conf_fallback), reverse=True)
    return [pt for pt in ranked if _box_score(pt, conf_fallback) > 0][:k]


def rematerialize_thumbs(
    detections: list[Detection],
    *,
    video_path: Path | None,
    out_dir: Path,
    force: bool = True,
) -> int:
    """Rebuild player crops from the match video.

    Picks the most chromatic jersey sample among the largest boxes so red kits
    are not drowned by a single grass/white frame. Offline dumps often point at
    /tmp/courtlens_thumbs or /kaggle/working which do not exist locally.
    """
    import cv2
    from app.services.team_assigner import _lab_chroma, jersey_colour_from_crop

    if video_path is None or not video_path.is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    need: list[tuple[Detection, list[dict]]] = []
    for d in detections:
        if d.class_name not in {"person", "referee"}:
            continue
        feats = d.features or {}
        path = feats.get("thumb_path")
        if (not force) and path and Path(str(path)).is_file():
            continue
        series = list(d.bbox_series or feats.get("bbox_series") or [])
        boxes = _top_boxes(series, float(d.confidence or 0.5), k=8)
        if boxes:
            need.append((d, boxes))
    if not need:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
    # One sequential pass, decoding only requested frames: per-box seeks decode from
    # the previous keyframe each time and took >30 min on a 10-min 60 fps match.
    wanted: dict[int, list[tuple[int, dict]]] = {}
    for i, (d, boxes) in enumerate(need):
        for pt in boxes:
            fi = pt.get("frame_idx")
            if fi is None:
                ms = pt.get("t_ms") or pt.get("ts_ms") or d.start_ms or 0
                fi = int(round(float(ms) * fps / 1000.0))
            wanted.setdefault(max(0, int(fi)), []).append((i, pt))
    best: dict[int, tuple[float, object, tuple[int, int]]] = {}
    last = max(wanted)
    try:
        for fi in range(last + 1):
            if fi not in wanted:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            for i, pt in wanted[fi]:
                x1, y1, x2, y2 = [int(pt[k]) for k in ("x1", "y1", "x2", "y2")]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 20 or y2 - y1 < 30:
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                out = jersey_colour_from_crop(crop)
                chroma = _lab_chroma(out[0]) if out is not None else 0.0
                area = float((x2 - x1) * (y2 - y1))
                # Prefer saturated jersey cloth; area is a tie-breaker only
                score = chroma * 10.0 + 0.0001 * area
                if score > best.get(i, (-1.0, None, (0, 0)))[0]:
                    best[i] = (score, crop, (x2 - x1, y2 - y1))
    finally:
        cap.release()

    written = 0
    for i, (_score, crop, wh) in best.items():
        d = need[i][0]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in d.track_key)
        dest = out_dir / f"{safe}.jpg"
        if not cv2.imwrite(str(dest), crop):
            continue
        feats = dict(d.features or {})
        feats["thumb_path"] = str(dest)
        feats["thumb_score"] = float(wh[0] * wh[1] * float(d.confidence or 0.5))
        d.features = feats
        written += 1
    return written


def _resolve_match_video(db: Session, match_id: UUID) -> Path | None:
    from app.models import VideoAsset
    from app.models.entities import VideoKind
    from app.services.storage import local_path

    asset = (
        db.query(VideoAsset)
        .filter(VideoAsset.match_id == match_id, VideoAsset.kind == VideoKind.original)
        .order_by(VideoAsset.created_at.desc())
        .first()
    )
    if asset and asset.object_key:
        p = local_path(asset.object_key)
        if p.is_file():
            return p
    return None


def inject_match_from_files(
    db: Session,
    match_id: UUID,
    detections_path: Path,
    events_path: Path | None = None,
    *,
    publish: bool = True,
    video_path: Path | None = None,
) -> dict:
    match = db.get(Match, match_id)
    if not match:
        raise RuntimeError(f"match {match_id} not found")

    detections = _load_detections(detections_path)
    video = video_path or _resolve_match_video(db, match_id)
    thumbs_dir = detections_path.parent / "track_thumbs"
    remade = rematerialize_thumbs(detections, video_path=video, out_dir=thumbs_dir)
    if remade:
        print(f"rematerialized {remade} thumbs → {thumbs_dir}")
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
                camera_label=CameraLabel(d.camera_label if d.camera_label in {"A", "B", "C", "D"} else "A"),
                track_key=d.track_key,
                start_ms=d.start_ms,
                end_ms=d.end_ms,
                confidence=d.confidence,
                features=feats,
            )
        )
    db.commit()
    run.status = JobStatus.succeeded
    # Keep the offline/Kaggle run summary (vision stats, timings, sync) for the Run stats panel
    summary_path = detections_path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        stages = summary.get("stages") or {}
        run.checkpoint = {
            "source": "offline",
            "track_count": len(detections),
            "vision_stats": (stages.get("detect_track") or {}).get("vision_stats"),
            "seconds": (stages.get("detect_track") or {}).get("seconds"),
            "wall_seconds": summary.get("wall_seconds"),
            "offsets": [{"camera": s["camera"], "offset_ms": s["offset_ms"], "confidence": s.get("confidence"),
                         "method": s.get("method")} for s in summary.get("sync") or []],
            "counts": summary.get("counts"),
            "settings": summary.get("settings"),
        }
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
        "assist": EventType.assist,
        "dribble": EventType.dribble,
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
