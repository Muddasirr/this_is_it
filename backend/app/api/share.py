from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models import HighlightClip, VideoAsset
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
