"""Render short evidence / highlight clips with FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from app.services.storage import upload_file


def cut_clip(src: Path, start_ms: int, end_ms: int, dest: Path) -> None:
    start_s = max(0, start_ms) / 1000.0
    duration_s = max(0.5, (end_ms - start_ms) / 1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(dest),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def render_and_store_clip(
    src: Path,
    object_key: str,
    start_ms: int,
    end_ms: int,
) -> str:
    with tempfile.TemporaryDirectory(prefix="clip-") as tmp:
        dest = Path(tmp) / f"{uuid4().hex}.mp4"
        cut_clip(src, start_ms, end_ms, dest)
        upload_file(dest, object_key)
    return object_key
