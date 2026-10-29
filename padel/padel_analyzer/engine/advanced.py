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


BACK_GLASS_M = 9.0  # contact this far from the net (≤ 1 m from the back glass): ball played off the wall


def shot_context(shots: list[dict], rallies: list[dict], player_pos: np.ndarray | None = None) -> None:
    """Annotate shots in place: serve side, return, direction, depth, zone, lob, off the wall, overhead kind
    (each estimated). player_pos: (n_frames, 4, 2) player floor positions, needed for the lob tag."""
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
            if pos:
                s["off_wall"] = abs(pos[1] - C.NET_Y) >= BACK_GLASS_M and j > 0
            if s.get("overhead") or s["type"] == "smash":
                # at the net it is a put-away (remate); from further back the controlled overheads (bandeja / víbora)
                s["overhead_kind"] = "smash" if s.get("zone") == "net" else "bandeja/víbora"
            # lob: both opponents at the net when it is hit and the reply comes from behind the service line
            if player_pos is not None and nxt and nxt["team"] != s["team"] and nxt.get("pos_m") and j > 0:
                opp = [2, 3] if s["team"] == 0 else [0, 1]
                f = min(s["frame"], len(player_pos) - 1)
                oy = player_pos[f, opp, 1]
                if not np.isnan(oy).any() and (C.zone(oy) == 0).all() and C.zone(nxt["pos_m"][1]) == 2:
                    s["lob"] = True


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
        "lobs": count("lob", True),
        "off_wall": count("off_wall", True),
        "overhead_kind": {k: count("overhead_kind", k) for k in ("smash", "bandeja/víbora")},
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


def _game_score(a: int, b: int, golden_point: bool) -> str:
    names = ["0", "15", "30", "40"]
    if a >= 3 and b >= 3 and not golden_point:
        return "40-40" if a == b else ("AD-40" if a > b else "40-AD")
    return f"{names[min(a, 3)]}-{names[min(b, 3)]}"


def outcome_stats(rep: dict, golden_point: bool = False) -> None:
    """Point-outcome stats from rally winners (model estimate or confirmed in review), in place.

    Winner / error: the last shot of a point won by the hitter's pair is a winner (or forced an error),
    of a lost point an error. Games follow the server: one pair serves a whole game, so a change of serving
    pair closes the game; the score is replayed with padel scoring (advantage, or golden point).
    """
    shots, rallies = rep["shots"], rep["rallies"]
    live = lambda r: [i for i in r["shots"] if not shots[i].get("deleted")]  # noqa: E731
    decided = [r for r in rallies if r.get("winner_team") is not None and live(r)]
    for p in rep["players"]:
        pid = p["id"]
        last = [r for r in decided if shots[live(r)[-1]]["player"] == pid]
        served = [r for r in decided if shots[live(r)[0]]["player"] == pid]
        p["stats"]["winners"] = sum(1 for r in last if r["winner_team"] == p["team"])
        p["stats"]["errors"] = sum(1 for r in last if r["winner_team"] != p["team"])
        for key, won in (("winners_by_type", True), ("errors_by_type", False)):
            ts = [shots[live(r)[-1]].get("type", "other") for r in last if (r["winner_team"] == p["team"]) == won]
            p["stats"][key] = {k: ts.count(k) for k in sorted(set(ts))}
        p["stats"]["points_served"] = len(served)
        p["stats"]["serve_points_won"] = sum(1 for r in served if r["winner_team"] == p["team"])
    for t in rep["teams"]:
        tid = t["id"]
        won = sum(1 for r in decided if r["winner_team"] == tid)
        t["stats"].update({
            "points_won": won,
            "points_won_pct": round(100 * won / len(decided), 1) if decided else None,
            "winners": sum(p["stats"]["winners"] for p in rep["players"] if p["team"] == tid),
            "errors": sum(p["stats"]["errors"] for p in rep["players"] if p["team"] == tid),
        })
        # points this pair finished with a shot from the net zone: won / lost
        net = [r for r in decided if shots[live(r)[-1]]["team"] == tid and shots[live(r)[-1]].get("zone") == "net"]
        t["stats"]["net_finishes"] = {"won": sum(r["winner_team"] == tid for r in net),
                                      "lost": sum(r["winner_team"] != tid for r in net)}
    # replay padel scoring over the decided points (assumes the recording starts at 0-0). Only confirmed
    # winners: the model is right on ~72 % of points (EVAL §9), too few for a reliable game score.
    decided = [r for r in decided if r.get("winner_source", "confirmed") == "confirmed"]
    log, sets, won_games = [], [], [0, 0]
    pts: list = []
    gs = [0, 0]  # games in the current set
    for r in decided:
        pts.append(r)
        a = sum(1 for x in pts if x["winner_team"] == 0)
        b = len(pts) - a
        tiebreak = gs == [6, 6]
        need = 7 if tiebreak else 4
        if max(a, b) >= need and (abs(a - b) >= 2 or (golden_point and not tiebreak)):
            gw = 0 if a > b else 1
            log.append({"points": [a, b], "game_winner": gw, "tiebreak": tiebreak, "rallies": [x["id"] for x in pts],
                        "first_hits": [shots[live(x)[0]]["team"] for x in pts]})
            won_games[gw] += 1
            gs[gw] += 1
            pts = []
            if (max(gs) >= 6 and abs(gs[0] - gs[1]) >= 2) or max(gs) == 7:
                sets.append(gs)
                gs = [0, 0]
    # the serve alternates every game: pick the order that agrees with the most first hits of the whole match
    # (a single detected first hit can be the return; the alternation makes the vote match-wide)
    games = log + ([{"first_hits": [shots[live(x)[0]]["team"] for x in pts], "rallies": [x["id"] for x in pts]}]
                   if pts else [])
    agree = [sum(f == (s0 + k) % 2 for k, g in enumerate(games) for f in g["first_hits"]) for s0 in (0, 1)]
    s0 = int(agree[1] > agree[0])
    by_id = {r["id"]: r for r in decided}
    # break points (the receiving pair can win the game on this point) and golden points (40-40, golden scoring)
    bp = {0: [0, 0], 1: [0, 0]}  # receiving team -> [break points, converted]
    golden = {0: [0, 0], 1: [0, 0]}  # team -> [golden points played, won]
    for k, g in enumerate(games):
        if g.get("tiebreak"):
            continue
        srv = (s0 + k) % 2
        rcv = 1 - srv
        cnt = [0, 0]
        for rid in g["rallies"]:
            w = by_id[rid]["winner_team"]
            rp, sp = cnt[rcv], cnt[srv]
            if rp >= 3 and (rp > sp or (golden_point and rp == sp == 3)):
                bp[rcv][0] += 1
                bp[rcv][1] += w == rcv
            if golden_point and cnt == [3, 3]:
                for tm in (0, 1):
                    golden[tm][0] += 1
                    golden[tm][1] += w == tm
            cnt[w] += 1
    for k, g in enumerate(games):
        for rid in g.pop("rallies"):
            by_id[rid]["server_team"] = (s0 + k) % 2
        del g["first_hits"]
        if "game_winner" in g:
            g["server_team"] = (s0 + k) % 2
            g["breaks"] = g["game_winner"] != g["server_team"]
    for t in rep["teams"]:
        tid = t["id"]
        srv = [r for r in decided if r["server_team"] == tid]
        ret = [r for r in decided if r["server_team"] != tid]
        t["stats"]["serve_points_won_pct"] = round(100 * sum(r["winner_team"] == tid for r in srv) / len(srv), 1) if srv else None
        t["stats"]["return_points_won_pct"] = round(100 * sum(r["winner_team"] == tid for r in ret) / len(ret), 1) if ret else None
    in_game = None
    if pts:
        a = sum(1 for x in pts if x["winner_team"] == 0)
        in_game = _game_score(a, len(pts) - a, golden_point)
    for t in rep["teams"]:
        t["stats"]["games_won"] = won_games[t["id"]]
        t["stats"]["service_games_held"] = sum(1 for g in log if g["server_team"] == t["id"] and g["game_winner"] == t["id"])
        t["stats"]["breaks"] = sum(1 for g in log if g["breaks"] and g["game_winner"] == t["id"])
        t["stats"]["break_points"] = {"won": bp[t["id"]][1], "of": bp[t["id"]][0]}
        t["stats"]["break_points_saved"] = {"saved": bp[1 - t["id"]][0] - bp[1 - t["id"]][1], "of": bp[1 - t["id"]][0]}
        if golden_point:
            t["stats"]["golden_points"] = {"won": golden[t["id"]][1], "of": golden[t["id"]][0]}
    rep["match"]["score"] = {"sets": sets + ([gs] if gs != [0, 0] or not sets else []), "current_game": in_game}
    rep["match"]["points_decided"] = len(decided)
    rep["match"]["games"] = log
    rep["match"]["games_won"] = won_games
    sources = {r.get("winner_source", "confirmed") for r in decided}
    rep["match"]["outcome_label"] = "confirmed" if sources == {"confirmed"} else "estimated" if decided else None
