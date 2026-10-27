from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_settings
from app.core.database import Base, engine
from app import models  # noqa: F401 — register metadata
from app.services.storage import ensure_bucket

logger = logging.getLogger(__name__)


_PG_PATCHES = [
    "ALTER TABLE players ADD COLUMN IF NOT EXISTS avatar_key VARCHAR(512)",
    # Enum values must be added outside a transaction (AUTOCOMMIT below)
    "ALTER TYPE eventtype ADD VALUE IF NOT EXISTS 'dribble'",
    *(f"ALTER TYPE eventtype ADD VALUE IF NOT EXISTS '{v}'" for v in (
        "interception", "save", "foul", "yellow_card", "red_card", "own_goal", "corner", "kick_in",
        "free_kick", "penalty")),
    "ALTER TYPE cameralabel ADD VALUE IF NOT EXISTS 'D'",
    # Per-camera sync: legacy rows were "B relative to A"
    "ALTER TABLE camera_syncs ADD COLUMN IF NOT EXISTS camera_label cameralabel NOT NULL DEFAULT 'B'",
    "ALTER TABLE match_shares ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ",
    "ALTER TABLE camera_sources ADD COLUMN IF NOT EXISTS calibration JSONB",
]


def _patch_postgres_schema() -> None:
    """Lightweight schema upgrades for existing DBs (create_all won't ALTER)."""
    from sqlalchemy import text

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for stmt in _PG_PATCHES:
            try:
                conn.execute(text(stmt))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Schema patch skipped (%s): %s", stmt, exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Lightweight schema upgrades for existing DBs (create_all won't ALTER)
    if engine.dialect.name == "postgresql":
        _patch_postgres_schema()
    try:
        ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Object storage not ready yet: %s", exc)
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Frame-Width", "X-Frame-Height"],  # calibration frame size (cross-origin)
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name}
