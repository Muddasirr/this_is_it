from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.deps import get_current_user, require_facility_member

from app.models import (
    CameraSource,
    CameraSync,
    FacilityUser,
    Match,
    MatchRoster,
    Player,
    ProcessingJob,
    User,
    VideoAsset,
)
from app.models.entities import CameraLabel, MatchStatus, VideoKind
from app.schemas import (
    CameraSyncOut,
    CameraSyncUpdate,
    MatchCreate,
    MatchOut,
    MatchRosterCreate,
    MatchRosterOut,
    ProcessingJobOut,
    VideoAssetOut,
    VideoUploadRequest,
    VideoUploadResponse,
)
from app.services.jobs import cancel_job, enqueue_processing_job, retry_job
from app.services.storage import object_key_for_original, presign_download, presign_upload

router = APIRouter()


def _get_match(db: Session, facility_id: UUID, match_id: UUID) -> Match:
    match = db.get(Match, match_id)
    if not match or match.facility_id != facility_id or match.status == MatchStatus.deleted:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.post("", response_model=MatchOut, status_code=status.HTTP_201_CREATED)
def create_match(
    facility_id: UUID,
    payload: MatchCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> Match:
    match = Match(
        facility_id=facility_id,
        title=payload.title,
        home_team_id=payload.home_team_id,
        away_team_id=payload.away_team_id,
        scheduled_date=payload.scheduled_date,
        notes=payload.notes,
        status=MatchStatus.awaiting_upload,
        created_by=user.id,
    )
    db.add(match)
    db.flush()
    for label in (CameraLabel.A, CameraLabel.B):
        db.add(CameraSource(match_id=match.id, label=label, description=f"Camera {label.value}"))
    db.commit()
    db.refresh(match)
    return match


@router.get("", response_model=list[MatchOut])
def list_matches(
    facility_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[Match]:
    return (
        db.query(Match)
        .filter(Match.facility_id == facility_id, Match.status != MatchStatus.deleted)
        .order_by(Match.created_at.desc())
        .all()
    )


@router.get("/{match_id}", response_model=MatchOut)
def get_match(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> Match:
    return _get_match(db, facility_id, match_id)


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_match(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> None:
    match = _get_match(db, facility_id, match_id)
    match.status = MatchStatus.deleted
    db.commit()


@router.post("/{match_id}/roster", response_model=MatchRosterOut, status_code=status.HTTP_201_CREATED)
def add_roster_entry(
    facility_id: UUID,
    match_id: UUID,
    payload: MatchRosterCreate,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> MatchRoster:
    match = _get_match(db, facility_id, match_id)
    player = db.get(Player, payload.player_id)
    if not player or player.facility_id != facility_id:
        raise HTTPException(status_code=400, detail="Player not in facility")
    entry = MatchRoster(
        match_id=match.id,
        player_id=payload.player_id,
        team_id=payload.team_id or player.team_id,
        shirt_number=payload.shirt_number,
        is_goalkeeper=payload.is_goalkeeper,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("/{match_id}/roster", response_model=list[MatchRosterOut])
def list_roster(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[MatchRoster]:
    _get_match(db, facility_id, match_id)
    return db.query(MatchRoster).filter(MatchRoster.match_id == match_id).all()


@router.post("/{match_id}/videos/upload-url", response_model=VideoUploadResponse)
def create_upload_url(
    facility_id: UUID,
    match_id: UUID,
    payload: VideoUploadRequest,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> VideoUploadResponse:
    match = _get_match(db, facility_id, match_id)
    camera = (
        db.query(CameraSource)
        .filter(CameraSource.match_id == match.id, CameraSource.label == payload.camera_label)
        .first()
    )
    if not camera:
        # A/B are created with the match; C/D appear when a court uploads them
        camera = CameraSource(
            match_id=match.id,
            label=payload.camera_label,
            description=f"Camera {payload.camera_label.value}",
        )
        db.add(camera)
        db.flush()

    key = object_key_for_original(facility_id, match.id, payload.camera_label.value, payload.filename)
    asset = VideoAsset(
        match_id=match.id,
        camera_source_id=camera.id,
        kind=VideoKind.original,
        object_key=key,
        content_type=payload.content_type,
        byte_size=payload.byte_size,
        validation_status="pending",
    )
    db.add(asset)
    if match.status == MatchStatus.awaiting_upload:
        match.status = MatchStatus.validating
    db.commit()
    db.refresh(asset)

    settings = get_settings()
    url = presign_upload(key, payload.content_type)
    return VideoUploadResponse(
        video_asset_id=asset.id,
        upload_url=url,
        object_key=key,
        expires_in=settings.signed_url_expire_seconds,
    )


@router.get("/{match_id}/videos", response_model=list[VideoAssetOut])
def list_videos(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[VideoAsset]:
    _get_match(db, facility_id, match_id)
    return db.query(VideoAsset).filter(VideoAsset.match_id == match_id).all()


@router.get("/{match_id}/videos/{asset_id}/playback-url")
def playback_url(
    facility_id: UUID,
    match_id: UUID,
    asset_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    _get_match(db, facility_id, match_id)
    asset = db.get(VideoAsset, asset_id)
    if not asset or asset.match_id != match_id:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"url": presign_download(asset.object_key), "expires_in": get_settings().signed_url_expire_seconds}


class CalibrationIn(BaseModel):
    # Clicked on the frame from GET …/cameras/{label}/frame, in that frame's pixels.
    # Order: near-left, far-left, far-right, near-right corner of the court.
    image_points: list[list[float]] = Field(min_length=4, max_length=40)
    # Court metres of each clicked point; omit for exactly the 4 corners. ≥ 8 points
    # (corners + halfway ends + centre-circle crossings) fit lens distortion too.
    court_points: list[list[float]] | None = None
    court_length_m: float = 40.0
    court_width_m: float = 20.0
    frame_w: int | None = None
    frame_h: int | None = None


def _camera(db: Session, match_id: UUID, label: CameraLabel) -> CameraSource:
    cam = db.query(CameraSource).filter(CameraSource.match_id == match_id, CameraSource.label == label).first()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera {label.value} not set up for this match")
    return cam


def _camera_video(db: Session, cam: CameraSource) -> VideoAsset | None:
    return (
        db.query(VideoAsset)
        .filter(VideoAsset.camera_source_id == cam.id, VideoAsset.kind == VideoKind.original)
        .order_by(VideoAsset.created_at.desc())
        .first()
    )


@router.get("/{match_id}/cameras")
def list_cameras(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[dict]:
    _get_match(db, facility_id, match_id)
    cams = db.query(CameraSource).filter(CameraSource.match_id == match_id).all()
    return [
        {
            "label": c.label.value,
            "has_video": _camera_video(db, c) is not None,
            "calibration": c.calibration,
        }
        for c in sorted(cams, key=lambda c: c.label.value)
    ]


@router.get("/{match_id}/cameras/{label}/frame")
def camera_frame(
    facility_id: UUID,
    match_id: UUID,
    label: CameraLabel,
    at: float = 0.2,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> Response:
    """JPEG still from the camera's original video (fraction `at` of its length) for corner clicks."""
    import tempfile
    from pathlib import Path

    import cv2

    from app.services.storage import download_to_path, local_path, use_local_storage

    _get_match(db, facility_id, match_id)
    asset = _camera_video(db, _camera(db, match_id, label))
    if not asset:
        raise HTTPException(status_code=404, detail="No video uploaded for this camera")
    with tempfile.TemporaryDirectory() as tmp:
        path = local_path(asset.object_key) if use_local_storage() else Path(tmp) / "v.mp4"
        if not use_local_storage():
            download_to_path(asset.object_key, path)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, min(0.95, at)) * n))
        ok, frame = cap.read()
        cap.release()
    if not ok:
        raise HTTPException(status_code=422, detail="Could not read a frame from this video")
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(content=jpg.tobytes(), media_type="image/jpeg",
                    headers={"X-Frame-Width": str(frame.shape[1]), "X-Frame-Height": str(frame.shape[0])})


@router.put("/{match_id}/cameras/{label}/calibration")
def set_calibration(
    facility_id: UUID,
    match_id: UUID,
    label: CameraLabel,
    payload: CalibrationIn,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    from app.services.court_calib import validate_calibration

    _get_match(db, facility_id, match_id)
    cam = _camera(db, match_id, label)
    try:
        cam.calibration = validate_calibration(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.commit()
    return {"label": label.value, "calibration": cam.calibration}


@router.delete("/{match_id}/cameras/{label}/calibration", status_code=status.HTTP_204_NO_CONTENT)
def clear_calibration(
    facility_id: UUID,
    match_id: UUID,
    label: CameraLabel,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> None:
    _get_match(db, facility_id, match_id)
    _camera(db, match_id, label).calibration = None
    db.commit()


@router.get("/{match_id}/run-stats")
def run_stats(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    """Processing-run quality numbers for the latest successful job (live worker or offline import)."""
    from app.models import ProcessingRun
    from app.models.entities import JobStage, JobStatus

    _get_match(db, facility_id, match_id)
    job = (
        db.query(ProcessingJob)
        # A processed match waits for staff review — that is a finished run too
        .filter(
            ProcessingJob.match_id == match_id,
            ProcessingJob.status.in_([JobStatus.succeeded, JobStatus.waiting_review]),
        )
        .order_by(ProcessingJob.created_at.desc())
        .first()
    )
    if not job:
        return {"available": False}
    runs = {r.stage: r for r in db.query(ProcessingRun).filter(ProcessingRun.job_id == job.id).all()}
    detect = (runs.get(JobStage.detect_track).checkpoint or {}) if JobStage.detect_track in runs else {}
    sync = (runs.get(JobStage.synchronize).checkpoint or {}) if JobStage.synchronize in runs else {}
    keep = (
        "model", "device", "tracker", "imgsz", "frames_processed", "frame_stride", "processing_fps", "realtime_factor",
        "video_seconds_probed", "person_tracks", "referee_tracks", "person_boxes", "ball_detections",
        "ball_coverage_pct", "reid_tracks", "jersey_ocr_tracks", "thumbs_written", "fallback_model",
    )
    cameras = []
    for v in detect.get("vision_stats") or []:
        row = {k: v.get(k) for k in keep if k in v}
        row["camera"] = v.get("camera_label", "A")
        if row.get("model"):
            row["model"] = str(row["model"]).rsplit("/", 1)[-1]
        if row.get("person_tracks") and row.get("person_boxes"):
            row["boxes_per_track"] = round(row["person_boxes"] / row["person_tracks"], 1)
        cameras.append(row)
    offsets = detect.get("offsets")
    if offsets is None and sync.get("offsets_ms"):
        offsets = [{"camera": k, "offset_ms": v} for k, v in sync["offsets_ms"].items() if k != "A"]
    return {
        "available": bool(cameras),
        "source": detect.get("source", "worker"),
        "job_finished_at": job.finished_at,
        "detect_seconds": detect.get("seconds"),
        "wall_seconds": detect.get("wall_seconds"),
        "track_count": detect.get("track_count"),
        "counts": detect.get("counts"),
        "cameras": cameras,
        "offsets": offsets or [],
    }


@router.put("/{match_id}/sync", response_model=CameraSyncOut)
def update_sync(
    facility_id: UUID,
    match_id: UUID,
    payload: CameraSyncUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> CameraSync:
    match = _get_match(db, facility_id, match_id)
    if payload.camera_label == CameraLabel.A:
        raise HTTPException(status_code=400, detail="Camera A is the reference timeline (offset 0)")
    previous = (
        db.query(CameraSync)
        .filter(CameraSync.match_id == match.id, CameraSync.camera_label == payload.camera_label)
        .order_by(CameraSync.created_at.desc())
        .first()
    )
    history = []
    if previous and previous.history:
        history = list(previous.history.get("corrections", []))
    if previous:
        history.append(
            {
                "offset_ms": previous.offset_ms,
                "method": previous.method,
                "is_manual": previous.is_manual,
            }
        )
    sync = CameraSync(
        match_id=match.id,
        camera_label=payload.camera_label,
        offset_ms=payload.offset_ms,
        confidence=1.0,
        method="manual",
        is_manual=True,
        notes=payload.notes,
        created_by=user.id,
        history={"corrections": history},
    )
    db.add(sync)
    db.commit()
    db.refresh(sync)
    return sync


@router.get("/{match_id}/sync", response_model=list[CameraSyncOut])
def get_sync(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[CameraSync]:
    """Latest offset per non-reference camera (B/C/D)."""
    _get_match(db, facility_id, match_id)
    rows = (
        db.query(CameraSync)
        .filter(CameraSync.match_id == match_id)
        .order_by(CameraSync.created_at.desc())
        .all()
    )
    latest: dict[CameraLabel, CameraSync] = {}
    for row in rows:
        latest.setdefault(row.camera_label, row)
    return [latest[k] for k in sorted(latest, key=lambda c: c.value)]


@router.post("/{match_id}/process", response_model=ProcessingJobOut, status_code=status.HTTP_202_ACCEPTED)
def start_processing(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> ProcessingJob:
    match = _get_match(db, facility_id, match_id)
    originals = (
        db.query(VideoAsset)
        .filter(VideoAsset.match_id == match.id, VideoAsset.kind == VideoKind.original)
        .all()
    )
    labels = set()
    for asset in originals:
        cam = db.get(CameraSource, asset.camera_source_id) if asset.camera_source_id else None
        if cam:
            labels.add(cam.label)
    settings = get_settings()
    if settings.allow_single_camera:
        if CameraLabel.A not in labels and CameraLabel.B not in labels:
            raise HTTPException(status_code=400, detail="Upload at least camera A or B")
    elif CameraLabel.A not in labels or CameraLabel.B not in labels:
        raise HTTPException(status_code=400, detail="Both camera A and B uploads are required")
    return enqueue_processing_job(db, match)


@router.get("/{match_id}/jobs", response_model=list[ProcessingJobOut])
def list_jobs(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[ProcessingJob]:
    _get_match(db, facility_id, match_id)
    return (
        db.query(ProcessingJob)
        .filter(ProcessingJob.match_id == match_id)
        .order_by(ProcessingJob.created_at.desc())
        .all()
    )


@router.post("/{match_id}/jobs/{job_id}/cancel", response_model=ProcessingJobOut)
def cancel_processing_job(
    facility_id: UUID,
    match_id: UUID,
    job_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> ProcessingJob:
    _get_match(db, facility_id, match_id)
    job = db.get(ProcessingJob, job_id)
    if not job or job.match_id != match_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return cancel_job(db, job)


@router.post("/{match_id}/jobs/retry", response_model=ProcessingJobOut, status_code=status.HTTP_202_ACCEPTED)
def retry_processing(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> ProcessingJob:
    match = _get_match(db, facility_id, match_id)
    return retry_job(db, match)
