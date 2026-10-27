"""Phase 3 — event review, clips, match stats, publish."""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_facility_member
from app.models import (
    CameraSource,
    EventCorrection,
    EventParticipant,
    EventReview,
    FacilityUser,
    HighlightClip,
    Match,
    MatchEvent,
    Player,
    User,
    VideoAsset,
)
from app.models.entities import EVENT_TYPE_BY_NAME, ConfirmationState, EventType, MatchStatus, VideoKind
from app.services.clips import render_and_store_clip
from app.services.storage import download_to_path, presign_download

router = APIRouter()


class EventOut(BaseModel):
    id: UUID
    event_type: str
    canonical_ts_ms: int
    confirmation: str
    confidence: float | None
    participants: list[dict]
    observations: list[dict]

    model_config = {"from_attributes": True}


class ReviewAction(BaseModel):
    action: str = Field(pattern="^(accept|reject|edit)$")
    notes: str | None = None
    event_type: str | None = None
    player_id: UUID | None = None
    canonical_ts_ms: int | None = None


class EventCreate(BaseModel):
    """Staff-added event (missed by the model). Stored confirmed — also futsal ground truth."""

    event_type: str  # any EVENT_TYPE_BY_NAME key (pass, shot, goal, foul, save, yellow_card …)
    canonical_ts_ms: int = Field(ge=0)
    player_id: UUID | None = None
    # Team-level events (corner, kick-in, foul when the player is unknown): which team it counts for
    team_id: UUID | None = None
    on_target: bool | None = None  # shots: staff call; else derived (followed by a goal or a save)
    notes: str | None = None

    @field_validator("event_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in EVENT_TYPE_BY_NAME:
            raise ValueError(f"event_type must be one of {', '.join(EVENT_TYPE_BY_NAME)}")
        return v


EVENT_TYPES = EVENT_TYPE_BY_NAME
# Role of the one player a staff-added event names; the rest use "primary"
PRIMARY_ROLE = {"pass": "passer", "shot": "shooter", "goal": "scorer", "assist": "passer", "dribble": "carrier",
                "interception": "won_by", "save": "goalkeeper", "own_goal": "scorer"}


class StatsOut(BaseModel):
    match_id: UUID
    status: str
    totals: dict
    by_player: list[dict]
    by_team: list[dict] | None = None
    pass_network: list[dict] | None = None
    heatmap: dict | None = None
    labels: dict


def _match(db: Session, facility_id: UUID, match_id: UUID) -> Match:
    match = db.get(Match, match_id)
    if not match or match.facility_id != facility_id or match.status == MatchStatus.deleted:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


def _event_type_value(et: EventType) -> str:
    return "pass" if et == EventType.pass_event else et.value


@router.get("/{match_id}/events", response_model=list[EventOut])
def list_events(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[EventOut]:
    _match(db, facility_id, match_id)
    events = (
        db.query(MatchEvent)
        .filter(MatchEvent.match_id == match_id, MatchEvent.is_canonical.is_(True))
        .order_by(MatchEvent.canonical_ts_ms)
        .all()
    )
    from app.models import EventObservation

    out: list[EventOut] = []
    for e in events:
        parts = db.query(EventParticipant).filter(EventParticipant.event_id == e.id).all()
        obs = db.query(EventObservation).filter(EventObservation.event_id == e.id).all()
        out.append(
            EventOut(
                id=e.id,
                event_type=_event_type_value(e.event_type),
                canonical_ts_ms=e.canonical_ts_ms,
                confirmation=e.confirmation.value,
                confidence=e.confidence,
                participants=[
                    {
                        "player_id": str(p.player_id) if p.player_id else None,
                        "track_id": str(p.track_id) if p.track_id else None,
                        "role": p.role,
                        "confidence": p.confidence,
                    }
                    for p in parts
                ],
                observations=[
                    {
                        "camera_label": o.camera_label.value,
                        "camera_ts_ms": o.camera_ts_ms,
                        "confidence": o.confidence,
                        "evidence_key": o.evidence_key,
                    }
                    for o in obs
                ],
            )
        )
    return out


@router.post("/{match_id}/events", response_model=EventOut, status_code=201)
def create_event(
    facility_id: UUID,
    match_id: UUID,
    payload: EventCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> EventOut:
    _match(db, facility_id, match_id)
    provenance = {"source": "staff", "user_id": str(user.id)}
    if payload.team_id is not None:
        provenance["team_id"] = str(payload.team_id)
    if payload.on_target is not None:
        provenance["on_target"] = payload.on_target
    event = MatchEvent(
        match_id=match_id,
        event_type=EVENT_TYPES[payload.event_type],
        canonical_ts_ms=payload.canonical_ts_ms,
        confirmation=ConfirmationState.confirmed,
        confidence=1.0,
        provenance=provenance,
        is_canonical=True,
    )
    db.add(event)
    db.flush()
    if payload.player_id is not None:
        if not db.get(Player, payload.player_id):
            raise HTTPException(status_code=404, detail="Player not found")
        db.add(EventParticipant(event_id=event.id, player_id=payload.player_id,
                                role=PRIMARY_ROLE.get(payload.event_type, "primary"), confidence=1.0))
    db.add(EventReview(event_id=event.id, reviewer_id=user.id, action="create", notes=payload.notes))
    db.commit()
    return _single_event(db, event)


@router.get("/{match_id}/events/export")
def export_ground_truth(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    """Confirmed events as ground truth (scripts/eval/score_events.py format)."""
    match = _match(db, facility_id, match_id)
    events = (
        db.query(MatchEvent)
        .filter(MatchEvent.match_id == match_id, MatchEvent.confirmation == ConfirmationState.confirmed)
        .order_by(MatchEvent.canonical_ts_ms)
        .all()
    )
    return {
        "video": match.title,
        "notes": "Staff-confirmed events (accepted model events + staff-added); rejected and unreviewed excluded.",
        "tolerance_ms": 3000,
        "events": [{"ts_ms": e.canonical_ts_ms, "type": _event_type_value(e.event_type),
                    "source": (e.provenance or {}).get("source", "model")} for e in events],
    }


@router.post("/{match_id}/events/{event_id}/review", response_model=EventOut)
def review_event(
    facility_id: UUID,
    match_id: UUID,
    event_id: UUID,
    payload: ReviewAction,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> EventOut:
    _match(db, facility_id, match_id)
    event = db.get(MatchEvent, event_id)
    if not event or event.match_id != match_id:
        raise HTTPException(status_code=404, detail="Event not found")

    old = {
        "confirmation": event.confirmation.value,
        "event_type": _event_type_value(event.event_type),
        "canonical_ts_ms": event.canonical_ts_ms,
    }

    if payload.action == "accept":
        event.confirmation = ConfirmationState.confirmed
    elif payload.action == "reject":
        event.confirmation = ConfirmationState.rejected
    elif payload.action == "edit":
        event.confirmation = ConfirmationState.confirmed
        if payload.event_type:
            if payload.event_type not in EVENT_TYPE_BY_NAME:
                raise HTTPException(status_code=400, detail="Invalid event_type")
            event.event_type = EVENT_TYPE_BY_NAME[payload.event_type]
        if payload.canonical_ts_ms is not None:
            event.canonical_ts_ms = payload.canonical_ts_ms
        if payload.player_id is not None:
            player = db.get(Player, payload.player_id)
            if not player or player.facility_id != facility_id:
                raise HTTPException(status_code=400, detail="Player not in facility")
            # replace primary participant
            db.query(EventParticipant).filter(EventParticipant.event_id == event.id).delete()
            db.add(
                EventParticipant(
                    event_id=event.id,
                    player_id=payload.player_id,
                    role="primary",
                    confidence=1.0,
                )
            )

    db.add(
        EventReview(
            event_id=event.id,
            reviewer_id=user.id,
            action=payload.action,
            notes=payload.notes,
        )
    )
    db.add(
        EventCorrection(
            event_id=event.id,
            reviewer_id=user.id,
            field_name="review",
            old_value=old,
            new_value={
                "confirmation": event.confirmation.value,
                "event_type": _event_type_value(event.event_type),
                "canonical_ts_ms": event.canonical_ts_ms,
                "player_id": str(payload.player_id) if payload.player_id else None,
            },
        )
    )
    db.commit()
    db.refresh(event)
    return _single_event(db, event)


def _single_event(db: Session, e: MatchEvent) -> EventOut:
    from app.models import EventObservation

    parts = db.query(EventParticipant).filter(EventParticipant.event_id == e.id).all()
    obs = db.query(EventObservation).filter(EventObservation.event_id == e.id).all()
    return EventOut(
        id=e.id,
        event_type=_event_type_value(e.event_type),
        canonical_ts_ms=e.canonical_ts_ms,
        confirmation=e.confirmation.value,
        confidence=e.confidence,
        participants=[
            {
                "player_id": str(p.player_id) if p.player_id else None,
                "track_id": str(p.track_id) if p.track_id else None,
                "role": p.role,
                "confidence": p.confidence,
            }
            for p in parts
        ],
        observations=[
            {
                "camera_label": o.camera_label.value,
                "camera_ts_ms": o.camera_ts_ms,
                "confidence": o.confidence,
                "evidence_key": o.evidence_key,
            }
            for o in obs
        ],
    )


@router.post("/{match_id}/events/{event_id}/clip")
def create_event_clip(
    facility_id: UUID,
    match_id: UUID,
    event_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    match = _match(db, facility_id, match_id)
    event = db.get(MatchEvent, event_id)
    if not event or event.match_id != match_id:
        raise HTTPException(status_code=404, detail="Event not found")

    asset = (
        db.query(VideoAsset)
        .filter(VideoAsset.match_id == match_id, VideoAsset.kind == VideoKind.original)
        .first()
    )
    if not asset:
        raise HTTPException(status_code=400, detail="No original video")

    cam = db.get(CameraSource, asset.camera_source_id) if asset.camera_source_id else None
    label = cam.label.value if cam else "A"
    start_ms = max(0, event.canonical_ts_ms - 3000)
    end_ms = event.canonical_ts_ms + 4000
    key = f"facilities/{facility_id}/matches/{match_id}/clips/{event_id}.mp4"

    with tempfile.TemporaryDirectory(prefix="evclip-") as tmp:
        src = Path(tmp) / "src.mp4"
        download_to_path(asset.object_key, src)
        render_and_store_clip(src, key, start_ms, end_ms)

    clip_asset = VideoAsset(
        match_id=match_id,
        camera_source_id=asset.camera_source_id,
        kind=VideoKind.clip,
        object_key=key,
        content_type="video/mp4",
        validation_status="passed",
    )
    db.add(clip_asset)
    db.flush()
    clip = HighlightClip(
        match_id=match_id,
        event_id=event.id,
        video_asset_id=clip_asset.id,
        title=f"{_event_type_value(event.event_type)} @ {event.canonical_ts_ms}ms",
        start_ms=start_ms,
        end_ms=end_ms,
        share_token=secrets.token_urlsafe(16),
    )
    db.add(clip)
    # attach evidence key on first observation
    from app.models import EventObservation

    obs = db.query(EventObservation).filter(EventObservation.event_id == event.id).first()
    if obs:
        obs.evidence_key = key
    db.commit()
    db.refresh(clip)
    return {
        "clip_id": str(clip.id),
        "share_token": clip.share_token,
        "playback_url": presign_download(key),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "camera": label,
    }


@router.get("/{match_id}/stats", response_model=StatsOut)
def match_stats(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> StatsOut:
    from app.services.match_stats import build_match_stats_cached

    match = _match(db, facility_id, match_id)
    built = build_match_stats_cached(db, match_id)
    return StatsOut(
        match_id=match.id,
        status=match.status.value,
        totals=built["totals"],
        by_player=built["by_player"],
        by_team=built.get("by_team"),
        pass_network=built.get("pass_network"),
        heatmap=built.get("heatmap"),
        labels=built["labels"],
    )


@router.post("/{match_id}/share/revoke")
def revoke_share(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    """Kill every active public link for this match (PRD: share links revocable)."""
    from datetime import UTC, datetime

    from app.models import MatchShare

    _match(db, facility_id, match_id)
    shares = (
        db.query(MatchShare)
        .filter(MatchShare.match_id == match_id, MatchShare.revoked_at.is_(None))
        .all()
    )
    for share in shares:
        share.revoked_at = datetime.now(UTC)
    db.commit()
    return {"match_id": str(match_id), "revoked": len(shares)}


@router.post("/{match_id}/publish")
def publish_match(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    from app.api.share import ensure_match_share
    from app.core.config import get_settings

    match = _match(db, facility_id, match_id)
    match.status = MatchStatus.published
    db.commit()
    share = ensure_match_share(db, match.id)
    # Front-end origin for the public report page
    report_path = f"/report/{share.share_token}"
    return {
        "match_id": str(match.id),
        "status": match.status.value,
        "share_token": share.share_token,
        "report_path": report_path,
        "api_report": f"{get_settings().api_public_url}/api/v1/share/matches/{share.share_token}",
    }


