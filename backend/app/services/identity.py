"""Phase 2 identity: propose roster links; staff confirms/corrects."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.models import MatchRoster, PlayerTrack, TrackIdentityLink
from app.models.entities import IdentityState


def propose_identity_links(db: Session, match_id: UUID, tracks: list[PlayerTrack]) -> list[TrackIdentityLink]:
    """Naive proposal: unused shirt numbers assigned round-robin by track start time.

    Real jersey OCR is deferred; this seeds the correction UI.
    """
    roster = db.query(MatchRoster).filter(MatchRoster.match_id == match_id).order_by(MatchRoster.shirt_number).all()
    if not roster:
        links = []
        for track in tracks:
            link = TrackIdentityLink(
                track_id=track.id,
                player_id=None,
                state=IdentityState.unknown,
                confidence=0.0,
                source="model",
            )
            db.add(link)
            links.append(link)
        db.commit()
        return links

    person_tracks = sorted(
        [t for t in tracks if (t.features or {}).get("class", "person") == "person"],
        key=lambda t: t.start_ms,
    )
    links: list[TrackIdentityLink] = []
    for i, track in enumerate(person_tracks):
        entry = roster[i % len(roster)]
        link = TrackIdentityLink(
            track_id=track.id,
            player_id=entry.player_id,
            state=IdentityState.proposed,
            confidence=0.25,
            source="model",
        )
        db.add(link)
        links.append(link)

    # non-person tracks → unknown
    for track in tracks:
        if track in person_tracks:
            continue
        link = TrackIdentityLink(
            track_id=track.id,
            player_id=None,
            state=IdentityState.unknown,
            confidence=0.0,
            source="model",
        )
        db.add(link)
        links.append(link)

    db.commit()
    return links


def set_track_identity(
    db: Session,
    track_id: UUID,
    player_id: UUID | None,
    user_id: UUID,
    state: IdentityState = IdentityState.confirmed,
) -> TrackIdentityLink:
    existing = (
        db.query(TrackIdentityLink)
        .filter(TrackIdentityLink.track_id == track_id)
        .order_by(TrackIdentityLink.created_at.desc())
        .first()
    )
    if existing and existing.state == IdentityState.confirmed and existing.player_id == player_id:
        return existing

    link = TrackIdentityLink(
        track_id=track_id,
        player_id=player_id,
        state=state if player_id else IdentityState.unknown,
        confidence=1.0 if state == IdentityState.confirmed else 0.0,
        source="staff",
        created_by=user_id,
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link
