"""Phase 0 feasibility API — upload a short public clip and run pretrained detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.deps import get_current_user
from app.models import User
from app.services.yolo_backend import run_feasibility_on_video

router = APIRouter()


@router.post("/feasibility")
async def run_feasibility(
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
) -> dict:
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return run_feasibility_on_video(path, camera_label="A")
    finally:
        Path(path).unlink(missing_ok=True)
