"""Vision on a Modal GPU (roadmap W1): VISION_BACKEND=modal.

The worker keeps the whole pipeline (identity, events, stats, review) and only
sends detect + track to the GPU — on the CPU worker that stage ran at ~0.02×
real time (65 s of video in 43 min). The video is uploaded to the "courtlens"
Modal volume, `courtlens.detect_video` (scripts/modal/courtlens_gpu.py, deployed
with `modal deploy`) runs the same YoloTrackBackend there, and returns the
detections plus a zip of the player thumbnails.

Credentials: the Modal CLI token (~/.modal.toml, mounted read-only into the
worker) or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.vision import Detection, EventCandidate, VisionBackend

logger = logging.getLogger(__name__)

MODAL_APP = "courtlens"
MODAL_VOLUME = "courtlens"
# Settings forwarded to the GPU run so it behaves exactly like a local run
FORWARD_ENV = ("YOLO_MODEL", "VISION_TARGET_FPS", "VISION_FRAME_STRIDE", "VISION_CONF", "VISION_IMGSZ",
               "VISION_IMGSZ_MAX", "VISION_MAX_FRAMES", "VISION_TRACKER", "VISION_REID", "VISION_TEAM_EMBED",
               "VISION_ROLE_FILTER", "VISION_JERSEY_OCR", "VISION_JERSEY_MODEL", "VISION_BALL_MODEL", "COURT_FORMAT")


class ModalTrackBackend(VisionBackend):
    name = "modal-yolo-botsort"
    version = "0.1.0"

    def __init__(self) -> None:
        self.last_run_stats: dict[str, Any] = {}
        self.spotted: dict[str, list[dict[str, Any]]] = {}  # camera → T-DEED events from the GPU run

    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        try:
            return self._remote(video_path, camera_label)
        except Exception as exc:  # noqa: BLE001 — no credits / outage / token: never fail the job for it
            logger.warning("Modal GPU detect failed (%s) — falling back to local YOLO on %s", exc,
                           get_settings().vision_device)
            from app.services.yolo_backend import YoloTrackBackend

            local = YoloTrackBackend()
            dets = local.detect_and_track(video_path, camera_label)
            self.last_run_stats = {**local.last_run_stats, "backend": "yolo-local-fallback", "modal_error": str(exc)[:300]}
            return dets

    def _remote(self, video_path: str, camera_label: str) -> list[Detection]:
        import modal

        job = uuid.uuid4().hex[:12]
        remote = f"/data/jobs/{job}/{camera_label}{Path(video_path).suffix or '.mp4'}"
        vol = modal.Volume.from_name(MODAL_VOLUME)
        with vol.batch_upload(force=True) as batch:
            batch.put_file(video_path, remote)
        env = {k: os.environ[k] for k in FORWARD_ENV if os.environ.get(k)}
        detect = modal.Function.from_name(MODAL_APP, "detect_video")
        logger.info("Modal detect_video job=%s camera=%s", job, camera_label)
        out = detect.remote(f"/mnt{remote}", camera_label, env)
        thumb_root = Path(get_settings().vision_thumb_dir or Path(tempfile.gettempdir()) / "courtlens_thumbs")
        thumb_dir = thumb_root / camera_label
        thumb_dir.mkdir(parents=True, exist_ok=True)
        if out.get("thumbs_zip"):
            with zipfile.ZipFile(io.BytesIO(out["thumbs_zip"])) as zf:
                zf.extractall(thumb_dir)
        dets = []
        for d in out["detections"]:
            feats = d.get("features") or {}
            if feats.get("thumb_path"):
                feats["thumb_path"] = str(thumb_dir / Path(feats["thumb_path"]).name)
            dets.append(Detection(**{**d, "features": feats}))
        self.last_run_stats = {**(out.get("stats") or {}), "backend": "modal", "modal_job": job}
        self.spotted[camera_label] = out.get("spotted") or []
        return dets

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import events_from_spotting, infer_event_candidates

        if self.spotted.get(camera_label):
            return events_from_spotting(self.spotted[camera_label], detections, camera_label)
        return infer_event_candidates(detections, camera_label)
