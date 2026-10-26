from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .settings import DB_URL

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
Session = sessionmaker(engine, expire_on_commit=False)


def _id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Camera(Base):
    """A fixed court camera: calibrate once, reuse for every match it records."""

    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_id)
    name: Mapped[str] = mapped_column(String(120))
    calibration: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[str] = mapped_column(String(12), primary_key=True, default=_id)
    title: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # uploaded -> calibrated -> queued -> processing -> done | failed
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    fps: Mapped[float] = mapped_column(Float, default=30.0)
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String(12), nullable=True)
    review: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # staff/player overrides

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "filename": self.filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status, "stage": self.stage, "progress": round(self.progress or 0, 3),
            "error": self.error, "fps": self.fps, "n_frames": self.n_frames, "width": self.width,
            "height": self.height, "duration_s": round(self.n_frames / self.fps, 1) if self.fps else None,
            "calibration": self.calibration, "camera_id": self.camera_id, "review": self.review or {},
        }


def init() -> None:
    Base.metadata.create_all(engine)
