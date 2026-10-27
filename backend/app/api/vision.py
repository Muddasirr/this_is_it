"""Phase 2 — tracks and roster identity correction."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_facility_member
from app.models import FacilityUser, Match, Player, PlayerTrack, TrackIdentityLink, User
from app.models.entities import IdentityState, MatchStatus
from app.services.avatars import thumb_url_for
from app.services.identity import set_track_identity

router = APIRouter()


class TrackOut(BaseModel):
    id: UUID
    camera_label: str
    track_key: str
    start_ms: int
    end_ms: int
    confidence: float | None
    class_name: str | None
    identity: dict | None
    thumb_url: str | None = None  # a crop of this track (staff identity board)

    model_config = {"from_attributes": True}


class IdentityUpdate(BaseModel):
    player_id: UUID | None = None
    state: IdentityState = IdentityState.confirmed


class BulkIdentityUpdate(BaseModel):
    """Move several tracks at once: merge a player into another (all its tracks), or mark crops
    as "not a player" (player_id null — referee, coach, crowd)."""

    track_ids: list[UUID]
    player_id: UUID | None = None


def _match(db: Session, facility_id: UUID, match_id: UUID) -> Match:
    match = db.get(Match, match_id)
    if not match or match.facility_id != facility_id or match.status == MatchStatus.deleted:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.get("/{match_id}/tracks", response_model=list[TrackOut])
def list_tracks(
    facility_id: UUID,
    match_id: UUID,
    db: Session = Depends(get_db),
    _: FacilityUser = Depends(require_facility_member),
) -> list[TrackOut]:
    _match(db, facility_id, match_id)
    tracks = db.query(PlayerTrack).filter(PlayerTrack.match_id == match_id).order_by(PlayerTrack.start_ms).all()
    out: list[TrackOut] = []
    for t in tracks:
        link = (
            db.query(TrackIdentityLink)
            .filter(TrackIdentityLink.track_id == t.id)
            .order_by(TrackIdentityLink.created_at.desc())
            .first()
        )
        identity = None
        if link:
            identity = {
                "link_id": str(link.id),
                "player_id": str(link.player_id) if link.player_id else None,
                "state": link.state.value,
                "confidence": link.confidence,
                "source": link.source,
            }
        out.append(
            TrackOut(
                id=t.id,
                camera_label=t.camera_label.value,
                track_key=t.track_key,
                start_ms=t.start_ms,
                end_ms=t.end_ms,
                confidence=t.confidence,
                class_name=(t.features or {}).get("class"),
                identity=identity,
                thumb_url=thumb_url_for(t),
            )
        )
    return out


@router.put("/{match_id}/tracks/{track_id}/identity", response_model=dict)
def update_identity(
    facility_id: UUID,
    match_id: UUID,
    track_id: UUID,
    payload: IdentityUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    _match(db, facility_id, match_id)
    track = db.get(PlayerTrack, track_id)
    if not track or track.match_id != match_id:
        raise HTTPException(status_code=404, detail="Track not found")
    if payload.player_id:
        player = db.get(Player, payload.player_id)
        if not player or player.facility_id != facility_id:
            raise HTTPException(status_code=400, detail="Player not in facility")
    link = set_track_identity(db, track_id, payload.player_id, user.id, payload.state)
    return {
        "link_id": str(link.id),
        "track_id": str(track_id),
        "player_id": str(link.player_id) if link.player_id else None,
        "state": link.state.value,
        "source": link.source,
    }


@router.put("/{match_id}/tracks/identity", response_model=dict)
def update_identities(
    facility_id: UUID,
    match_id: UUID,
    payload: BulkIdentityUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: FacilityUser = Depends(require_facility_member),
) -> dict:
    _match(db, facility_id, match_id)
    if payload.player_id:
        player = db.get(Player, payload.player_id)
        if not player or player.facility_id != facility_id:
            raise HTTPException(status_code=400, detail="Player not in facility")
    tracks = db.query(PlayerTrack).filter(PlayerTrack.id.in_(payload.track_ids), PlayerTrack.match_id == match_id).all()
    if len(tracks) != len(set(payload.track_ids)):
        raise HTTPException(status_code=404, detail="Track not found in this match")
    for t in tracks:
        set_track_identity(db, t.id, payload.player_id, user.id)
    return {"updated": len(tracks), "player_id": str(payload.player_id) if payload.player_id else None}
