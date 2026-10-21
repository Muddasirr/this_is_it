"""Attach match-frame player thumbnails to roster players."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Match, Player, PlayerTrack, TrackIdentityLink
from app.services.storage import presign_download, upload_file

logger = logging.getLogger(__name__)


def attach_player_avatars(db: Session, match: Match, tracks: list[PlayerTrack]) -> int:
    """Upload best in-match crop per player → player.avatar_key. Returns count updated."""
    if not tracks:
        return 0

    # Latest identity link per track
    track_ids = [t.id for t in tracks]
    links = (
        db.query(TrackIdentityLink)
        .filter(TrackIdentityLink.track_id.in_(track_ids))
        .order_by(TrackIdentityLink.created_at.desc())
        .all()
    )
    player_of: dict[UUID, UUID] = {}
    for link in links:
        if link.track_id in player_of:
            continue
        if link.player_id:
            player_of[link.track_id] = link.player_id

    # Best thumb per player. Never reuse the same source image for two players
    # (that made MANAV #23 appear as both Kit A #2 and Kit B #2).
    candidates: list[tuple[float, UUID, Path]] = []
    for track in tracks:
        pid = player_of.get(track.id)
        if not pid:
            continue
        feats = track.features or {}
        path_str = feats.get("thumb_path")
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        # Skip remapped offline kit*.jpg stamps shared across many tracks
        if "thumbs/kit" in path_str.replace("\\", "/").lower() and path_str.count("kit") >= 1:
            name = path.name.lower()
            if name.startswith("kita_") or name.startswith("kitb_") or name.startswith("kit"):
                # Still allow, but down-rank heavily so unique track crops win
                score_scale = 0.05
            else:
                score_scale = 1.0
        else:
            score_scale = 1.0
        score = (
            float(feats.get("thumb_score") or track.confidence or 0.5)
            * max(1.0, (track.end_ms - track.start_ms) / 1000.0)
            * score_scale
        )
        candidates.append((score, pid, path))

    candidates.sort(key=lambda x: -x[0])
    best: dict[UUID, Path] = {}
    used_paths: set[str] = set()
    for score, pid, path in candidates:
        if pid in best:
            continue
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in used_paths:
            continue
        best[pid] = path
        used_paths.add(resolved)

    updated = 0
    for pid, path in best.items():
        player = db.get(Player, pid)
        if not player:
            continue
        key = f"avatars/{match.facility_id}/{pid}.jpg"
        try:
            upload_file(path, key, content_type="image/jpeg")
            player.avatar_key = key
            updated += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Avatar upload failed for %s: %s", pid, exc)

    if updated:
        db.commit()
    return updated


def avatar_url_for(player: Player | None) -> str | None:
    if not player or not player.avatar_key:
        return None
    try:
        return presign_download(player.avatar_key)
    except Exception:  # noqa: BLE001
        return None
