"""Cross-camera event fuse (A/B/C/D): one real-world event → one canonical event.

Each camera's events engine already emits one candidate per action it sees.
Fusing therefore only merges observations from *different* cameras that
(1) have the same event type, (2) land within `bucket_ms` of each other on the
canonical timeline (camera_ts + sync offset, Camera A = reference), and
(3) beyond `same_moment_ms`, do not name different resolved players. Inside
that window two angles disagreeing on the player is attribution noise (seen on
Kaggle: one pass, passer Kit A #8 on B vs Kit B #2 on A, 120 ms apart), not two
events. Raw tracker IDs are never compared across cameras.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _get(c: Any, name: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(name, default)
    return getattr(c, name, default)


def canonical_ts_ms(candidate: Any, offsets_ms: dict[str, int] | None = None) -> int:
    cam = str(_get(candidate, "camera_label") or "A")
    return int(_get(candidate, "camera_ts_ms") or 0) + int((offsets_ms or {}).get(cam, 0))


def group_event_candidates(
    candidates: list[Any],
    *,
    bucket_ms: int = 1500,
    offsets_ms: dict[str, int] | None = None,
    player_of: Callable[[str], Any] | None = None,
    same_moment_ms: int = 400,
) -> list[list[Any]]:
    """Group candidates into canonical events; each group is that event's observations.

    `offsets_ms[camera]` maps camera time onto Camera A time. `player_of(track_key)`
    returns the resolved player (or None) used to keep two different players'
    passes apart when they happen in the same second on two angles.
    """
    rows = []
    for c in candidates:
        et = _get(c, "event_type")
        if not et or _get(c, "camera_ts_ms") is None:
            continue
        keys = _get(c, "track_keys") or []
        player = player_of(keys[0]) if (player_of and keys) else None
        rows.append((canonical_ts_ms(c, offsets_ms), str(et), str(_get(c, "camera_label") or "A"), player, c))
    rows.sort(key=lambda r: r[0])

    groups: list[dict] = []
    open_by_type: dict[str, list[dict]] = {}
    for ts, et, cam, player, c in rows:
        target = None
        for g in open_by_type.get(et, []):
            if ts - g["anchor_ts"] > bucket_ms:
                continue
            if cam in g["cameras"]:
                continue  # same camera ⇒ the events engine already split them
            disagree = player is not None and g["players"] and player not in g["players"]
            if disagree and ts - g["anchor_ts"] > same_moment_ms:
                continue
            target = g
            break
        if target is None:
            target = {"anchor_ts": ts, "cameras": set(), "players": set(), "obs": []}
            groups.append(target)
            open_by_type.setdefault(et, []).append(target)
        target["cameras"].add(cam)
        if player is not None:
            target["players"].add(player)
        target["obs"].append(c)
        # Drop groups that can no longer accept members
        open_by_type[et] = [g for g in open_by_type[et] if ts - g["anchor_ts"] <= bucket_ms]
    return [g["obs"] for g in groups]


def fuse_event_candidates(
    candidates: list[Any],
    *,
    bucket_ms: int = 1500,
    offsets_ms: dict[str, int] | None = None,
    player_of: Callable[[str], Any] | None = None,
) -> list[Any]:
    """Highest-confidence observation of each canonical event."""
    groups = group_event_candidates(
        candidates, bucket_ms=bucket_ms, offsets_ms=offsets_ms, player_of=player_of
    )
    return [max(g, key=lambda c: float(_get(c, "confidence") or 0.0)) for g in groups]
