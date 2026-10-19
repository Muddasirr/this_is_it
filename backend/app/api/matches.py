from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
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
        raise HTTPException(status_code=400, detail="Camera source missing")

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
    previous = (
        db.query(CameraSync).filter(CameraSync.match_id == match.id).order_by(CameraSync.created_at.desc()).first()
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


@router.get("/{match_id}/sync", response_model=CameraSyncOut | None)
def get_sync(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> CameraSync | None:
    _get_match(db, facility_id, match_id)
    return (
        db.query(CameraSync)
        .filter(CameraSync.match_id == match_id)
        .order_by(CameraSync.created_at.desc())
        .first()
    )


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
