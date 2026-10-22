from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.entities import CameraLabel, JobStage, JobStatus, MatchStatus, VideoKind


class MatchCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    home_team_id: UUID | None = None
    away_team_id: UUID | None = None
    scheduled_date: date | None = None
    notes: str | None = None


class MatchOut(BaseModel):
    id: UUID
    facility_id: UUID
    home_team_id: UUID | None
    away_team_id: UUID | None
    title: str
    scheduled_date: date | None
    status: MatchStatus
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MatchRosterCreate(BaseModel):
    player_id: UUID
    team_id: UUID | None = None
    shirt_number: int = Field(ge=0, le=99)
    is_goalkeeper: bool = False


class MatchRosterOut(BaseModel):
    id: UUID
    match_id: UUID
    player_id: UUID
    team_id: UUID | None
    shirt_number: int
    is_goalkeeper: bool

    model_config = {"from_attributes": True}


class VideoUploadRequest(BaseModel):
    camera_label: CameraLabel
    filename: str
    content_type: str = "video/mp4"
    byte_size: int | None = None


class VideoUploadResponse(BaseModel):
    video_asset_id: UUID
    upload_url: str
    object_key: str
    expires_in: int


class VideoAssetOut(BaseModel):
    id: UUID
    match_id: UUID
    camera_source_id: UUID | None
    kind: VideoKind
    object_key: str
    content_type: str
    byte_size: int | None
    duration_ms: int | None
    width: int | None
    height: int | None
    frame_rate: float | None
    codec: str | None
    validation_status: str
    validation_errors: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class CameraSyncUpdate(BaseModel):
    # Offset of this camera onto Camera A's timeline: canonical = camera_ts + offset_ms
    camera_label: CameraLabel = CameraLabel.B
    offset_ms: int
    notes: str | None = None


class CameraSyncOut(BaseModel):
    id: UUID
    match_id: UUID
    camera_label: CameraLabel
    offset_ms: int
    confidence: float | None
    method: str
    is_manual: bool
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProcessingJobOut(BaseModel):
    id: UUID
    match_id: UUID
    status: JobStatus
    stage: JobStage | None
    progress_pct: int
    error_message: str | None
    cost_cents: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}
