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


SYNC_MAX_LAG_MS = 60_000
SYNC_WINDOW_S = 300  # analyse the first 5 min — enough for a whistle / kick-off
SYNC_MIN_CONFIDENCE = 0.3  # below this the UI asks staff to set the offset


def xcorr_lag(ref, other, rate_hz: float, max_lag_ms: int = SYNC_MAX_LAG_MS) -> tuple[int, float]:
    """Lag (ms) where `other` best matches `ref`, plus a peak-sharpness confidence.

    Positive lag ⇒ the moment at other[t] appears at ref[t + lag], i.e.
    canonical (ref) time = other time + lag. Signals are z-scored first.
    """
    import numpy as np

    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if a.size < 8 or b.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0, 0.0
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    max_lag = min(int(max_lag_ms * rate_hz / 1000.0), n // 2 - 1)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([corr[: max_lag + 1], corr[n - max_lag :]])
    # Normalise by overlap so long lags are not penalised for fewer samples
    overlap = np.minimum(a.size - np.maximum(lags, 0), b.size + np.minimum(lags, 0)).clip(min=1)
    vals = vals / overlap
    i = int(np.argmax(vals))
    peak = float(vals[i])
    # Sharpness: peak vs best value outside ±0.5 s of it
    guard = max(1, int(0.5 * rate_hz))
    mask = np.abs(lags - lags[i]) > guard
    runner_up = float(vals[mask].max()) if mask.any() else 0.0
    conf = 0.0 if peak <= 0 else max(0.0, min(1.0, (peak - max(runner_up, 0.0)) / peak))
    return int(round(lags[i] * 1000.0 / rate_hz)), round(conf, 3)


def _audio_envelope(path: str | Path, rate_hz: int = 100, window_s: int = SYNC_WINDOW_S):
    """Onset envelope (positive energy flux) at `rate_hz` from the first audio track."""
    import numpy as np

    sr = 8000
    cmd = [
        "ffmpeg", "-v", "quiet", "-t", str(window_s), "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    hop = sr // rate_hz
    if pcm.size < hop * 16:
        return None
    frames = pcm[: (pcm.size // hop) * hop].reshape(-1, hop)
    energy = np.log1p(np.sqrt((frames**2).mean(axis=1)))
    return np.clip(np.diff(energy, prepend=energy[0]), 0, None)


def _motion_energy(path: str | Path, rate_hz: float = 10.0, window_s: int = SYNC_WINDOW_S):
    """Mean abs frame difference at ~`rate_hz` — play bursts look alike from every angle."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
    step = max(1, int(round(fps / rate_hz)))
    out: list[float] = []
    times: list[float] = []
    prev = None
    idx = 0
    limit = int(window_s * fps)
    try:
        while idx < limit:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.float32)
                out.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
                times.append(idx / fps)
                prev = g
            idx += 1
    finally:
        cap.release()
    if len(out) < 16:
        return None
    # Resample onto a fixed grid so cameras with different FPS line up
    grid = np.arange(0.0, times[-1], 1.0 / rate_hz)
    return np.interp(grid, times, out)


def estimate_offset_ms(
    ref_path: str | Path,
    other_path: str | Path,
    *,
    ref_has_audio: bool = False,
    other_has_audio: bool = False,
) -> tuple[int, float, str]:
    """Estimate `other` camera offset relative to the reference camera (A).

    Returns (offset_ms, confidence, method) with canonical_ts = camera_ts + offset_ms.
    Audio onset cross-correlation first (whistles / kicks hit every mic at once);
    motion-energy cross-correlation for silent CCTV. Low confidence ⇒ staff fix.
    """
    if ref_has_audio and other_has_audio:
        try:
            a = _audio_envelope(ref_path)
            b = _audio_envelope(other_path)
            if a is not None and b is not None:
                lag, conf = xcorr_lag(a, b, 100.0)
                if conf >= SYNC_MIN_CONFIDENCE:
                    return lag, conf, "audio_xcorr"
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        ma = _motion_energy(ref_path)
        mb = _motion_energy(other_path)
    except Exception:  # noqa: BLE001 — cv2 decode issues must not fail the job
        ma = mb = None
    if ma is not None and mb is not None:
        lag, conf = xcorr_lag(ma, mb, 10.0)
        return lag, conf, "motion_xcorr"
    return 0, 0.0, "none"


def download_to_temp(content: bytes, suffix: str = ".mp4") -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)
