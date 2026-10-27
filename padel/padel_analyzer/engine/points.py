"""Who won the point: P(the pair that hit the last shot won the rally).

A point ends with the last shot being a winner (or forcing an error) — the hitter's pair wins — or an error
(net, out, into the own glass) — it loses. The hitter's side is known; what decides it is what the ball and
the opponents do next. Features (all relative to the last hitter, so the model is camera- and team-agnostic):
  last shot      zone, stroke probabilities, overhead / volley, hit score, rally length
  ball after it  how long it stays visible, where its last sighting projects (own / opponent half, lateral),
                 whether it crossed the net line in the image
  opponents      how far they moved toward the ball in the next 1.5 s, their distance to the net
Labels come from broadcast scoreboards (docs/EVAL.md §9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import court as C

MODEL = Path(__file__).resolve().parents[2] / "models" / "point_lgbm.txt"
STROKES = ["serve", "forehand", "backhand", "smash", "other"]
FEATURES = [
    "n_shots", "duration_s", "hit_score", "overhead", "volley", "zone", "dist_net",
    *[f"p_{c}" for c in STROKES],
    "ball_vis_s", "ball_last_dy", "ball_last_x_out", "ball_crossed", "ball_seen_after",
    "opp_move_m", "opp_net_dist", "own_net_dist", "next_gap_s", "far_side",
]


def _sign_toward_opponent(y_hitter: float) -> float:
    """+1 if the opponent half is at larger y (hitter on the far side, y < 10)."""
    return 1.0 if y_hitter < C.NET_Y else -1.0


def rally_features(rally: dict, shots: list[dict], ball_xy: np.ndarray, pos_m: np.ndarray, calib, fps: float,
                   seg_end: int, next_start: int | None) -> np.ndarray:
    """One row of FEATURES for a rally (nan where unknown)."""
    idx = [i for i in rally["shots"] if not shots[i].get("deleted")]
    x = np.full(len(FEATURES), np.nan)
    if not idx:
        return x
    last = shots[idx[-1]]
    f = int(last["frame"])
    n = len(ball_xy)
    v = dict.fromkeys(FEATURES, np.nan)
    v["n_shots"] = len(idx)
    v["duration_s"] = (last["frame"] - shots[idx[0]]["frame"]) / fps
    v["hit_score"] = last.get("hit_score", np.nan)
    v["overhead"] = float(bool(last.get("overhead")))
    v["volley"] = float(bool(last.get("volley")))
    probs = last.get("type_probs") or {}
    for c in STROKES:
        v[f"p_{c}"] = probs.get(c, float(last.get("type") == c))
    hp = last.get("pos_m")
    team = last["team"]
    if hp is None:
        p = pos_m[min(f, len(pos_m) - 1), last["player"]]
        hp = None if np.isnan(p).any() else [float(p[0]), float(p[1])]
    if hp is None:
        return np.array([v[k] for k in FEATURES], float)
    sgn = _sign_toward_opponent(hp[1])
    v["far_side"] = float(hp[1] < C.NET_Y)
    v["zone"] = float(C.zone(hp[1]))
    v["dist_net"] = abs(hp[1] - C.NET_Y)
    # ball after the last hit, until the next rally / scene cut / 3 s
    end = min(n, seg_end, f + int(3 * fps), next_start if next_start is not None else n)
    w = ball_xy[f + 2:end] if end > f + 2 else np.zeros((0, 2))
    seen = ~np.isnan(w[:, 0]) if len(w) else np.zeros(0, bool)
    v["ball_seen_after"] = float(seen.mean()) if len(seen) else 0.0
    if seen.any():
        k = np.where(seen)[0]
        v["ball_vis_s"] = (k[-1] + 2) / fps
        m = calib.to_m(w[k])
        d = (m[:, 1] - C.NET_Y) * sgn  # > 0: opponent half
        v["ball_last_dy"] = float(d[-1])
        v["ball_last_x_out"] = float(max(0.0, -m[-1, 0], m[-1, 0] - C.WIDTH))
        v["ball_crossed"] = float((d > 0.5).any())
    else:
        v["ball_vis_s"] = 0.0
    # opponents: movement in the next 1.5 s, distance to the net at the hit
    opp = [2, 3] if team == 0 else [0, 1]
    own = [0, 1] if team == 0 else [2, 3]
    f2 = min(len(pos_m) - 1, f + int(1.5 * fps), end - 1 if end > f else f)
    a, b = pos_m[min(f, len(pos_m) - 1), opp], pos_m[f2, opp]
    mv = np.linalg.norm(b - a, axis=1)
    v["opp_move_m"] = float(np.nanmean(mv)) if not np.isnan(mv).all() else np.nan
    v["opp_net_dist"] = float(np.nanmean(np.abs(a[:, 1] - C.NET_Y))) if not np.isnan(a[:, 1]).all() else np.nan
    o = pos_m[min(f, len(pos_m) - 1), own]
    v["own_net_dist"] = float(np.nanmean(np.abs(o[:, 1] - C.NET_Y))) if not np.isnan(o[:, 1]).all() else np.nan
    v["next_gap_s"] = ((next_start - f) / fps) if next_start is not None else np.nan
    return np.array([v[k] for k in FEATURES], float)


class PointModel:
    def __init__(self, path: Path = MODEL):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(path))
        meta = path.with_suffix(".json")
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    @staticmethod
    def available(path: Path = MODEL) -> bool:
        return path.exists()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(last hitter's pair won) per row."""
        return self.booster.predict(X) if len(X) else np.zeros(0)


def assign_winners(rallies: list[dict], shots: list[dict], X: np.ndarray, model: PointModel) -> None:
    """Fill winner_team (estimated) on rallies without a confirmed winner."""
    p = model.predict(X)
    for r, pr in zip(rallies, p):
        idx = [i for i in r["shots"] if not shots[i].get("deleted")]
        if not idx or r.get("winner_source") == "confirmed" or np.isnan(pr):
            continue
        t = shots[idx[-1]]["team"]
        r["winner_team"] = t if pr >= 0.5 else 1 - t
        r["winner_source"] = "estimated"
        r["winner_conf"] = round(float(max(pr, 1 - pr)), 2)
