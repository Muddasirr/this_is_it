"""Player / pair / match statistics from person-indexed tracks, hits, strokes and rallies."""
from __future__ import annotations

import numpy as np

from . import court as C

SPRINT_MS = 4.0
HEAT_BIN = 0.5


def _speed(pos: np.ndarray, fps: float) -> np.ndarray:
    """(n,) m/s from a smoothed (n, 2) track, central difference over 0.2 s."""
    k = max(1, int(round(0.1 * fps)))
    v = np.full(len(pos), np.nan)
    if len(pos) > 2 * k:
        v[k:-k] = np.linalg.norm(pos[2 * k:] - pos[:-2 * k], axis=1) / (2 * k / fps)
    return v


def _distance(pos: np.ndarray, fps: float, max_speed: float = 8.0) -> float:
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    d = np.where(np.isnan(d) | (d > 1.5 * max_speed / fps), 0.0, d)
    return float(d.sum())


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask, False].astype(int)
    starts = np.where(np.diff(m) == 1)[0]
    ends = np.where(np.diff(m) == -1)[0]
    return list(zip(starts, ends))


def folded_heatmap(pos: np.ndarray) -> list[list[float]]:
    """Occupancy on the player's own half, seen from behind the player: rows = distance from the net
    (0..10 m), cols = left..right (far-side x mirrored)."""
    ok = ~np.isnan(pos[:, 0])
    p = pos[ok]
    if not len(p):
        return []
    far = p[:, 1] < C.NET_Y
    x = np.where(far, C.WIDTH - p[:, 0], p[:, 0])
    d = np.abs(p[:, 1] - C.NET_Y)
    H, _, _ = np.histogram2d(np.clip(d, 0, 9.999), np.clip(x, 0, 9.999),
                             bins=[int(10 / HEAT_BIN), int(10 / HEAT_BIN)], range=[[0, 10], [0, 10]])
    H = H / H.sum()
    return np.round(H, 5).tolist()


def player_stats(pos: np.ndarray, fps: float) -> dict:
    present = ~np.isnan(pos[:, 0])
    t = present.sum() / fps
    v = _speed(pos, fps)
    vv = v[present & ~np.isnan(v)]
    vv = vv[vv < 8.5]
    dist = _distance(pos, fps)
    sprints = [r for r in _runs(np.nan_to_num(v) > SPRINT_MS) if (r[1] - r[0]) / fps >= 0.5]
    z = C.zone(pos[present, 1]) if present.any() else np.zeros(0)
    return {
        "time_tracked_s": round(float(t), 1),
        "distance_m": round(dist, 1),
        "avg_speed_kmh": round(float(dist / t * 3.6), 2) if t > 0 else None,
        "max_speed_kmh": round(float(np.percentile(vv, 99.5) * 3.6), 1) if len(vv) > 10 else None,
        "sprints": len(sprints),
        "zone_pct": {
            "net": round(float((z == 0).mean() * 100), 1) if len(z) else None,
            "transition": round(float((z == 1).mean() * 100), 1) if len(z) else None,
            "back": round(float((z == 2).mean() * 100), 1) if len(z) else None,
        },
    }


def pair_stats(pa: np.ndarray, pb: np.ndarray) -> dict:
    both = ~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0])
    if not both.any():
        return {}
    a, b = pa[both], pb[both]
    za, zb = C.zone(a[:, 1]), C.zone(b[:, 1])
    return {
        "both_at_net_pct": round(float(((za == 0) & (zb == 0)).mean() * 100), 1),
        "both_back_pct": round(float(((za == 2) & (zb == 2)).mean() * 100), 1),
        "avg_partner_distance_m": round(float(np.linalg.norm(a - b, axis=1).mean()), 2),
        "avg_lateral_spacing_m": round(float(np.abs(a[:, 0] - b[:, 0]).mean()), 2),
        "avg_depth_offset_m": round(float(np.abs(a[:, 1] - b[:, 1]).mean()), 2),
    }


def shot_ground_speed(hit_pos: list[np.ndarray], frames: list[int], rally_of: list[int], fps: float) -> list[float | None]:
    """Ball ground speed estimate per shot: hitter position -> next hitter position over flight time."""
    out: list[float | None] = []
    for i in range(len(frames)):
        j = i + 1
        if j >= len(frames) or rally_of[j] != rally_of[i]:
            out.append(None)
            continue
        dt = (frames[j] - frames[i]) / fps
        a, b = hit_pos[i], hit_pos[j]
        if dt <= 0.15 or np.isnan(a).any() or np.isnan(b).any():
            out.append(None)
            continue
        v = float(np.linalg.norm(b - a) / dt * 3.6)
        out.append(round(v, 1) if v < 200 else None)
    return out
