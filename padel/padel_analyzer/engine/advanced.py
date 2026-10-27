"""Padel tactical stats derived from measured positions + detected shots / rallies.

Positions (metres) are measured; anything that depends on a detected shot is estimated (EVAL: hits F1 0.94
fused, hitter 0.90 / team 0.94, stroke type 0.79). Shot direction / depth use where the *next* hitter stood
(the ball's landing spot is not observable in 3D from one camera), so they describe where the opponent had
to play the ball from.
"""
from __future__ import annotations

import numpy as np

from . import court as C

RALLY_BUCKETS = [(1, 3), (4, 6), (7, 9), (10, 10**6)]


def _side_of_court(x: float, far: bool) -> str:
    """'left' / 'right' as seen by the player facing the net (far players face the camera)."""
    left = x < C.WIDTH / 2
    if far:
        left = not left
    return "left" if left else "right"


def shot_context(shots: list[dict], rallies: list[dict]) -> None:
    """Annotate shots in place: serve side, return, direction, depth, zone (each estimated)."""
    for r in rallies:
        idx = [i for i in r["shots"] if not shots[i].get("deleted")]
        for j, i in enumerate(idx):
            s = shots[i]
            s["rally_index"] = j
            s["is_return"] = j == 1
            pos = s.get("pos_m")
            if pos:
                far = pos[1] < C.NET_Y
                s["zone"] = ["net", "transition", "back"][int(C.zone(pos[1]))]
                if s["type"] == "serve" or j == 0:
                    s["serve_side"] = _side_of_court(pos[0], far)
            nxt = shots[idx[j + 1]] if j + 1 < len(idx) else None
            if pos and nxt and nxt.get("pos_m") and nxt["team"] != s["team"]:
                q = nxt["pos_m"]
                dx = abs(q[0] - C.WIDTH / 2)
                if dx < 1.25:
                    s["direction"] = "centre"
                else:
                    same_x_half = (pos[0] < C.WIDTH / 2) == (q[0] < C.WIDTH / 2)
                    # opposite ends of the court: the same x-half is down the line, the other is crosscourt
                    s["direction"] = "down the line" if same_x_half else "crosscourt"
                s["depth"] = "deep" if C.zone(q[1]) == 2 else ("mid" if C.zone(q[1]) == 1 else "short")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.r_[False, mask, False].astype(int)
    return list(zip(np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0]))


def net_approaches(y: np.ndarray, fps: float, min_s: float = 1.0) -> int:
    """Times the player moved from behind the service line into the net zone and stayed ≥ min_s."""
    z = C.zone(np.where(np.isnan(y), C.NET_Y + 8, y))
    z = np.where(np.isnan(y), -1, z)
    count, was_back = 0, False
    for a, b in _runs(z == 0):
        if (b - a) / fps < min_s:
            continue
        before = z[max(0, a - int(3 * fps)):a]
        was_back = bool((before == 2).any())
        count += was_back
    return count


def player_extra(p: int, shots: list[dict], rallies: list[dict], pos: np.ndarray, fps: float,
                 in_play: np.ndarray) -> dict:
    mine = [s for s in shots if s["player"] == p and not s.get("deleted")]
    team = 0 if p < 2 else 1
    y = pos[:, 1]
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    d = np.where(np.isnan(d) | (d > 0.5), 0.0, d)
    play_min = in_play.sum() / fps / 60
    rallies_hit = sum(1 for r in rallies if any(shots[i]["player"] == p for i in r["shots"]))
    count = lambda key, val: sum(1 for s in mine if s.get(key) == val)  # noqa: E731
    serves = [s for s in mine if s.get("rally_index") == 0]
    ok = ~np.isnan(pos[:, 0])
    return {
        "serves": len(serves),
        "serve_side": {"left": sum(1 for s in serves if s.get("serve_side") == "left"),
                       "right": sum(1 for s in serves if s.get("serve_side") == "right")},
        "returns": sum(1 for s in mine if s.get("is_return")),
        "shots_by_zone": {z: count("zone", z) for z in ("net", "transition", "back")},
        "direction": {k: count("direction", k) for k in ("crosscourt", "down the line", "centre")},
        "depth": {k: count("depth", k) for k in ("deep", "mid", "short")},
        "last_shot_of_rally": sum(1 for r in rallies if r["shots"] and shots[r["shots"][-1]]["player"] == p),
        "rally_participation_pct": round(100 * rallies_hit / max(1, len(rallies)), 1),
        "net_approaches": net_approaches(y, fps),
        "distance_in_play_m": round(float(d[in_play[1:]].sum()), 1),
        "work_rate_m_per_min": round(float(d[in_play[1:]].sum() / play_min), 1) if play_min > 0 else None,
        "avg_position_m": [round(float(np.nanmean(pos[ok, 0])), 2), round(float(np.nanmean(np.abs(y[ok] - C.NET_Y))), 2)]
        if ok.any() else None,  # [x, distance from net]
        "team": team,
    }


def pair_extra(pa: np.ndarray, pb: np.ndarray, shots: list[dict], rallies: list[dict], team: int,
               fps: float) -> dict:
    both = ~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0])
    out: dict = {}
    if both.any():
        za, zb = C.zone(pa[both, 1]), C.zone(pb[both, 1])
        atk = (za == 0) & (zb == 0)
        dfn = (za == 2) & (zb == 2)
        out["formation_pct"] = {"attack_both_net": round(float(atk.mean() * 100), 1),
                                "defence_both_back": round(float(dfn.mean() * 100), 1),
                                "split_one_up_one_back": round(float((~atk & ~dfn).mean() * 100), 1)}
        out["net_control_pct"] = round(float(((za == 0) | (zb == 0)).mean() * 100), 1)
    served = [r for r in rallies if r["shots"] and shots[r["shots"][0]]["team"] == team]
    out["rallies_served"] = len(served)
    # how long after its own serve the serving pair has both players inside the net zone
    t_net = []
    for r in served:
        a = int(r["start_t"] * fps)
        b = min(len(pa), int(r["end_t"] * fps))
        for f in range(a, b):
            if not (np.isnan(pa[f, 1]) or np.isnan(pb[f, 1])) and C.zone(pa[f, 1]) == 0 and C.zone(pb[f, 1]) == 0:
                t_net.append((f - a) / fps)
                break
    out["avg_time_to_net_after_serve_s"] = round(float(np.mean(t_net)), 2) if t_net else None
    tm = [s for s in shots if s["team"] == team and not s.get("deleted")]
    out["direction"] = {k: sum(1 for s in tm if s.get("direction") == k) for k in ("crosscourt", "down the line", "centre")}
    return out


def match_extra(rallies: list[dict], shots: list[dict], duration_s: float, fps: float, n: int) -> tuple[dict, np.ndarray]:
    in_play = np.zeros(n, bool)
    for r in rallies:
        in_play[int(r["start_t"] * fps):int(r["end_t"] * fps) + 1] = True
    t_play = in_play.sum() / fps
    lens = [r["n_shots"] for r in rallies]
    live = [s for s in shots if not s.get("deleted")]
    return {
        "time_in_play_s": round(float(t_play), 1),
        "time_in_play_pct": round(100 * t_play / max(duration_s, 1e-9), 1),
        "shots_per_min_in_play": round(len(live) / (t_play / 60), 1) if t_play > 0 else None,
        "rally_length_distribution": {f"{a}-{b}" if b < 10**6 else f"{a}+": sum(1 for x in lens if a <= x <= b)
                                      for a, b in RALLY_BUCKETS},
        "direction_mix": {k: sum(1 for s in live if s.get("direction") == k) for k in ("crosscourt", "down the line", "centre")},
    }, in_play
