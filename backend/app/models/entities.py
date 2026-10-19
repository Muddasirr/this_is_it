from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class FacilityRole(str, enum.Enum):
    owner = "owner"
    staff = "staff"
    reviewer = "reviewer"


class MatchStatus(str, enum.Enum):
    draft = "draft"
    awaiting_upload = "awaiting_upload"
    validating = "validating"
    ready_to_process = "ready_to_process"
    processing = "processing"
    needs_review = "needs_review"
    published = "published"
    failed = "failed"
    deleted = "deleted"


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    waiting_review = "waiting_review"


class JobStage(str, enum.Enum):
    ingest = "ingest"
    normalize = "normalize"
    synchronize = "synchronize"
    detect_track = "detect_track"
    resolve_identity = "resolve_identity"
    infer_events = "infer_events"
    fuse = "fuse"
    review_gate = "review_gate"
    publish = "publish"


class VideoKind(str, enum.Enum):
    original = "original"
    proxy = "proxy"
    clip = "clip"
    artifact = "artifact"


class CameraLabel(str, enum.Enum):
    A = "A"
    B = "B"


class EventType(str, enum.Enum):
    goal = "goal"
    shot = "shot"
    pass_event = "pass"
    assist = "assist"


class ConfirmationState(str, enum.Enum):
    estimated = "estimated"
    confirmed = "confirmed"
    rejected = "rejected"
    unavailable = "unavailable"


class IdentityState(str, enum.Enum):
    proposed = "proposed"
    confirmed = "confirmed"
    unknown = "unknown"
    uncertain = "uncertain"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    memberships: Mapped[list[FacilityUser]] = relationship(back_populates="user")


class Facility(Base):
    __tablename__ = "facilities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    users: Mapped[list[FacilityUser]] = relationship(back_populates="facility")
    teams: Mapped[list[Team]] = relationship(back_populates="facility")
    matches: Mapped[list[Match]] = relationship(back_populates="facility")
    retention_policy: Mapped[RetentionPolicy | None] = relationship(back_populates="facility", uselist=False)


class FacilityUser(Base):
    __tablename__ = "facility_users"
    __table_args__ = (UniqueConstraint("facility_id", "user_id", name="uq_facility_user"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[FacilityRole] = mapped_column(Enum(FacilityRole), default=FacilityRole.staff)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    facility: Mapped[Facility] = relationship(back_populates="users")
    user: Mapped[User] = relationship(back_populates="memberships")


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    primary_color: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    facility: Mapped[Facility] = relationship(back_populates="teams")
    players: Mapped[list[Player]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"))
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id", ondelete="SET NULL"), nullable=True)
    display_name: Mapped[str] = mapped_column(String(120))
    default_shirt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    team: Mapped[Team | None] = relationship(back_populates="players")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"), index=True)
    home_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    away_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    scheduled_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[MatchStatus] = mapped_column(Enum(MatchStatus), default=MatchStatus.draft, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    facility: Mapped[Facility] = relationship(back_populates="matches")
    camera_sources: Mapped[list[CameraSource]] = relationship(back_populates="match")
    video_assets: Mapped[list[VideoAsset]] = relationship(back_populates="match")
    rosters: Mapped[list[MatchRoster]] = relationship(back_populates="match")
    jobs: Mapped[list[ProcessingJob]] = relationship(back_populates="match")
    syncs: Mapped[list[CameraSync]] = relationship(back_populates="match")


class MatchRoster(Base):
    __tablename__ = "match_rosters"
    __table_args__ = (UniqueConstraint("match_id", "player_id", name="uq_match_player"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id", ondelete="CASCADE"))
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    shirt_number: Mapped[int] = mapped_column(Integer)
    is_goalkeeper: Mapped[bool] = mapped_column(Boolean, default=False)

    match: Mapped[Match] = relationship(back_populates="rosters")
    player: Mapped[Player] = relationship()


class CameraSource(Base):
    __tablename__ = "camera_sources"
    __table_args__ = (UniqueConstraint("match_id", "label", name="uq_match_camera_label"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    label: Mapped[CameraLabel] = mapped_column(Enum(CameraLabel))
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    match: Mapped[Match] = relationship(back_populates="camera_sources")


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    camera_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("camera_sources.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[VideoKind] = mapped_column(Enum(VideoKind), default=VideoKind.original)
    object_key: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default="video/mp4")
    byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    has_audio: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(32), default="pending")
    validation_errors: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped[Match] = relationship(back_populates="video_assets")


class CameraSync(Base):
    __tablename__ = "camera_syncs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    offset_ms: Mapped[int] = mapped_column(Integer, default=0)  # B relative to A
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    method: Mapped[str] = mapped_column(String(64), default="manual")
    is_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    history: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    match: Mapped[Match] = relationship(back_populates="syncs")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str] = mapped_column(String(64))  # detection | tracking | events
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued, index=True)
    stage: Mapped[JobStage | None] = mapped_column(Enum(JobStage), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    match: Mapped[Match] = relationship(back_populates="jobs")
    runs: Mapped[list[ProcessingRun]] = relationship(back_populates="job")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("processing_jobs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[JobStage] = mapped_column(Enum(JobStage))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    checkpoint: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    logs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped[ProcessingJob] = relationship(back_populates="runs")


class PlayerTrack(Base):
    __tablename__ = "player_tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    processing_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("processing_runs.id", ondelete="CASCADE"))
    camera_label: Mapped[CameraLabel] = mapped_column(Enum(CameraLabel))
    track_key: Mapped[str] = mapped_column(String(64))  # camera-scoped track id
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class TrackIdentityLink(Base):
    __tablename__ = "track_identity_links"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("player_tracks.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    state: Mapped[IdentityState] = mapped_column(Enum(IdentityState), default=IdentityState.proposed)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="model")  # model | staff
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MatchEvent(Base):
    __tablename__ = "match_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    processing_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("processing_runs.id"), nullable=True)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType, values_callable=lambda x: [e.value for e in x]))
    canonical_ts_ms: Mapped[int] = mapped_column(Integer)
    confirmation: Mapped[ConfirmationState] = mapped_column(
        Enum(ConfirmationState), default=ConfirmationState.estimated
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    provenance: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EventObservation(Base):
    __tablename__ = "event_observations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("match_events.id", ondelete="CASCADE"), index=True)
    camera_label: Mapped[CameraLabel] = mapped_column(Enum(CameraLabel))
    camera_ts_ms: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class EventParticipant(Base):
    __tablename__ = "event_participants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("match_events.id", ondelete="CASCADE"), index=True)
    player_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    track_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("player_tracks.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(64))  # scorer | shooter | passer | receiver
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class EventReview(Base):
    __tablename__ = "event_reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("match_events.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(32))  # accept | reject | edit
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EventCorrection(Base):
    __tablename__ = "event_corrections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("match_events.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    field_name: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class HighlightClip(Base):
    __tablename__ = "highlight_clips"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    event_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("match_events.id"), nullable=True)
    video_asset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("video_assets.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    share_token: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"), index=True)
    match_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("matches.id"), nullable=True)
    category: Mapped[str] = mapped_column(String(64))  # gpu_seconds | storage_gb_days | egress | review_minutes
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    unit_cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetentionPolicy(Base):
    __tablename__ = "retention_policies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    facility_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("facilities.id", ondelete="CASCADE"), unique=True
    )
    original_days: Mapped[int] = mapped_column(Integer, default=30)
    proxy_days: Mapped[int] = mapped_column(Integer, default=90)
    clip_days: Mapped[int] = mapped_column(Integer, default=365)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    facility: Mapped[Facility] = relationship(back_populates="retention_policy")
