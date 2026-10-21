"""Cross-camera event fuse (A/B/C). Same type + time bucket → one canonical key."""

from __future__ import annotations

from typing import Any


def fuse_event_candidates(
    candidates: list[Any],
    *,
    bucket_ms: int = 1500,
) -> list[Any]:
    """Dedupe EventCandidate-like objects across cameras.

    Keeps highest-confidence observation per (event_type, time_bucket, primary track).
    """
    best: dict[tuple, Any] = {}
    for c in candidates:
        et = getattr(c, "event_type", None) or (c.get("event_type") if isinstance(c, dict) else None)
        ts = getattr(c, "camera_ts_ms", None)
        if ts is None and isinstance(c, dict):
            ts = c.get("camera_ts_ms")
        keys = getattr(c, "track_keys", None)
        if keys is None and isinstance(c, dict):
            keys = c.get("track_keys") or []
        conf = getattr(c, "confidence", None)
        if conf is None and isinstance(c, dict):
            conf = c.get("confidence", 0)
        if not et or ts is None:
            continue
        primary = (keys[0] if keys else None) or ""
        # Strip camera prefix from track for cross-cam identity (A-p1 / B-p1 → weak)
        primary_norm = primary.split("-", 1)[-1] if primary else ""
        bucket = int(ts) // max(1, bucket_ms)
        key = (et, bucket, primary_norm)
        prev = best.get(key)
        prev_conf = getattr(prev, "confidence", None) if prev is not None else None
        if prev is None:
            best[key] = c
        elif prev_conf is not None and conf is not None and conf > prev_conf:
            best[key] = c
        elif isinstance(prev, dict) and conf is not None and conf > float(prev.get("confidence") or 0):
            best[key] = c
    return list(best.values())
