"""FFmpeg helpers for Phase 0 validation / normalization / sync."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class MediaProbeError(Exception):
    pass


def probe_video(path: str | Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise MediaProbeError("ffprobe not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaProbeError(exc.stderr or "ffprobe failed") from exc

    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not video_stream:
        raise MediaProbeError("No video stream found")

    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    num, den = fps_raw.split("/")
    fps = float(num) / float(den) if float(den) else 0.0

    return {
        "duration_ms": int(duration * 1000),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "frame_rate": round(fps, 3),
        "codec": video_stream.get("codec_name"),
        "has_audio": audio_stream is not None,
        "format_name": fmt.get("format_name"),
        "bit_rate": int(fmt.get("bit_rate") or 0) or None,
    }


def validate_probe(probe: dict[str, Any], min_duration_ms: int = 60_000) -> list[str]:
    errors: list[str] = []
    if probe["duration_ms"] < min_duration_ms:
        errors.append(f"Duration under {min_duration_ms // 1000}s — likely incomplete recording")
    if probe["width"] < 640 or probe["height"] < 360:
        errors.append("Resolution below minimum validation floor")
    if probe["frame_rate"] < 10:
        errors.append("Frame rate too low for reliable ball tracking")
    if probe["codec"] and probe["codec"] not in {"h264", "hevc", "h265", "mpeg4", "av1"}:
        errors.append(f"Unsupported codec: {probe['codec']}")
    return errors


def make_proxy(src: str | Path, dst: str | Path, max_width: int = 960) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"scale='min({max_width},iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def estimate_offset_ms_stub(duration_a_ms: int, duration_b_ms: int) -> tuple[int, float, str]:
    """Placeholder sync until Phase 0 footage tests land audio/visual alignment.

    Returns (offset_ms B relative to A, confidence, method).
    """
    # Prefer zero when durations are similar; flag low confidence for staff correction.
    delta = abs(duration_a_ms - duration_b_ms)
    if delta < 500:
        return 0, 0.35, "duration_heuristic"
    return 0, 0.15, "duration_heuristic_low"


def download_to_temp(content: bytes, suffix: str = ".mp4") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)
