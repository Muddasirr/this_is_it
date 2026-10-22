import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models import Facility, HighlightClip, Match, MatchShare, VideoAsset
from app.models.entities import MatchStatus
from app.services.match_stats import build_match_stats
from app.services.storage import presign_download

router = APIRouter()


@router.get("/clips/{share_token}")
def shared_clip(share_token: str, db: Session = Depends(get_db)) -> dict:
    clip = db.query(HighlightClip).filter(HighlightClip.share_token == share_token).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    asset = db.get(VideoAsset, clip.video_asset_id) if clip.video_asset_id else None
    if not asset:
        raise HTTPException(status_code=404, detail="Clip asset missing")
    return {
        "title": clip.title,
        "start_ms": clip.start_ms,
        "end_ms": clip.end_ms,
        "playback_url": presign_download(asset.object_key),
    }


@router.get("/matches/{share_token}")
def shared_match_report(share_token: str, db: Session = Depends(get_db)) -> dict:
    share = db.query(MatchShare).filter(MatchShare.share_token == share_token).first()
    if not share:
        raise HTTPException(status_code=404, detail="Report not found")
    match = db.get(Match, share.match_id)
    if not match or match.status == MatchStatus.deleted:
        raise HTTPException(status_code=404, detail="Match not found")
    facility = db.get(Facility, match.facility_id)
    stats = build_match_stats(db, match.id)
    # Include per-player heatmaps (image-plane, estimated) for share report
    players = []
    for row in stats["by_player"]:
        players.append(
            {
                "display_name": row["display_name"],
                "player_id": row.get("player_id"),
                "shirt_number": row.get("shirt_number"),
                "avatar_url": row.get("avatar_url"),
                "goal": row["goal"],
                "shot": row["shot"],
                "pass": row["pass"],
                "assist": row.get("assist", 0),
                "dribble": row.get("dribble", 0),
                "pass_received": row.get("pass_received", 0),
                "possession_pct": row.get("possession_pct", 0),
                "touches": row.get("touches", 0),
                "distance": row.get("distance", 0),
                "minutes": row.get("minutes", 0),
                "avg_speed": row.get("avg_speed", 0),
                "conversion_pct": row.get("conversion_pct"),
                "attribution": row.get("attribution", "estimated"),
                "heatmap": row.get("heatmap"),
                "possession_heatmap": row.get("possession_heatmap"),
            }
        )
    return {
        "title": match.title,
        "facility_name": facility.name if facility else "Court",
        "status": match.status.value,
        "totals": stats["totals"],
        "by_player": players,
        "by_team": stats.get("by_team") or [],
        "heatmap": {
            "width": stats["heatmap"]["width"],
            "height": stats["heatmap"]["height"],
            "match": stats["heatmap"]["match"],
            "match_possession": stats["heatmap"].get("match_possession"),
            "note": stats["heatmap"].get("note"),
            "by_player": True,
        },
        "labels": stats["labels"],
    }


def ensure_match_share(db: Session, match_id) -> MatchShare:
    existing = db.query(MatchShare).filter(MatchShare.match_id == match_id).first()
    if existing:
        return existing
    share = MatchShare(match_id=match_id, share_token=secrets.token_urlsafe(16))
    db.add(share)
    db.commit()
    db.refresh(share)
    return share
