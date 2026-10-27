"""Kernel: point labels from broadcast scoreboards -> (1) pair identity across ends, (2) point-winner model.

Inputs: PT100 videos + held-out detection caches (both finals), padel-scoreboard states, the Kuwait 2025 clip's
detections / report / hand-read points (padel-kuwait-small), production models.
  1. identity: orientation (which pair plays the far end) per segment, old vs new linking, against the truth
     from the scoreboard serve dot + where the serve was hit.
  2. points: one row per scoreboard point change -> engine.points features, label = last hitter's pair won.
     Leave-one-match-out over F / M / Kuwait; production model trained on all three.
Output: /kaggle/working/out/points.json, models/point_lgbm.{txt,json}
"""
from __future__ import annotations

import glob
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

OUT = Path("/kaggle/working/out")
REPO = Path("/kaggle/working/padel")


def first(p: str) -> Path | None:
    h = sorted(glob.glob(p, recursive=True))
    return Path(h[0]) if h else None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "lightgbm"], check=False)
    code = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(code, REPO)
    models = REPO / "models"
    models.mkdir(exist_ok=True)
    wanted = {
        "pose.pt": "/kaggle/input/**/models/pose_all/best.pt",
        "hit_lgbm_video.txt": "/kaggle/input/**/padel-eval/out/models/hit_lgbm.txt",
        "hit_lgbm_video.json": "/kaggle/input/**/padel-eval/out/models/hit_lgbm.json",
        "hit_lgbm.txt": "/kaggle/input/**/padel-fusion/out/models/hit_lgbm.txt",
        "hit_lgbm.json": "/kaggle/input/**/padel-fusion/out/models/hit_lgbm.json",
        "shot_lgbm.txt": "/kaggle/input/**/out/models/shot_lgbm.txt",
        "stroke_rgb.pt": "/kaggle/input/**/stroke_rgb.pt",
    }
    for name, pat in wanted.items():
        src = first(pat)
        if src:
            shutil.copy(src, models / name)
    labels = first("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json").parent
    (REPO / "data" / "pt100").mkdir(parents=True, exist_ok=True)
    if not (REPO / "data" / "pt100" / "labels").exists():
        shutil.copytree(labels, REPO / "data" / "pt100" / "labels")
    sys.path.insert(0, str(REPO))
    log("models", sorted(p.name for p in models.iterdir()))


# ---- scoreboard parsing ---------------------------------------------------------------------------------
PTS = {"0": 0, "O": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def parse_row(tokens: list[str]) -> tuple[list[int], int] | None:
    """[name tokens..., set games..., points] -> (games per set, point index)."""
    nums = [t.replace("O", "0") if re.fullmatch(r"[0-9O]+", t) else t for t in tokens]
    nums = [t for t in nums if re.fullmatch(r"[0-9]+|AD|A", t)]
    if not nums or nums[-1] not in PTS:
        return None
    games = []
    for t in nums[:-1]:
        if t.isdigit() and len(t) == 1:
            games.append(int(t))
    return games, PTS[nums[-1]]


def transitions(states: list[dict]) -> list[dict]:
    """Consecutive readable states that differ by exactly one point -> {start, end (frames), winner_row, server_row}."""
    rd = []
    for s in states:
        rows = (s.get("p") or {}).get("rows") or []
        if len(rows) != 2:
            continue
        a, b = parse_row(rows[0]), parse_row(rows[1])
        if a is None or b is None or len(a[0]) != len(b[0]):  # no games column yet at 0-0 in the first game
            continue
        dot = s.get("dot") or [0, 0]
        srv = 0 if dot[0] > max(30, 2 * dot[1]) else 1 if dot[1] > max(30, 2 * dot[0]) else None
        rd.append({"start": s["start"], "end": s["end"], "games": (a[0], b[0]), "pts": (a[1], b[1]), "srv": srv})
    out = []
    for x, y in zip(rd, rd[1:]):
        if x["games"] == y["games"]:
            d = (y["pts"][0] - x["pts"][0], y["pts"][1] - x["pts"][1])
            win = 0 if d == (1, 0) else 1 if d == (0, 1) else None
            # deuce back from advantage: AD-40 -> 40-40 means the other row won
            if x["pts"] == (4, 3) and y["pts"] == (3, 3):
                win = 1
            if x["pts"] == (3, 4) and y["pts"] == (3, 3):
                win = 0
        else:
            ga, gb = x["games"], y["games"]
            dg = [sum(gb[0]) - sum(ga[0]), sum(gb[1]) - sum(ga[1])]
            win = 0 if dg == [1, 0] and y["pts"] == (0, 0) and x["pts"][0] >= 3 else \
                1 if dg == [0, 1] and y["pts"] == (0, 0) and x["pts"][1] >= 3 else None
        if win is not None:
            out.append({"from": x["start"], "to": y["start"], "winner_row": win, "server_row": x["srv"]})
    return out


def old_link_identity(segments, det_of, hists, n):
    """team[f, slot], person[f, slot]. Segment 0: far pair = team 0, slots 0..3 = persons 0..3.
    Later segments: the side whose kit matches team 0's kit is team 0; partners by individual colour."""
    team = np.zeros((n, 4), int)
    person = np.tile(np.arange(4), (n, 1))
    team[:, 2:] = 1
    if not hists or len(segments) < 2:
        return team, person
    def slot_hist(a, b, s):
        hs = [hists[(f, det_of[f, s])] for f in range(a, b) if det_of[f, s] >= 0 and (f, det_of[f, s]) in hists]
        return np.mean(hs, 0) if hs else None
    proto_person: dict[int, np.ndarray] = {}
    a, b = segments[0]
    for s in range(4):
        h = slot_hist(a, b, s)
        if h is not None:
            proto_person[s] = h
    for a, b in segments[1:]:
        hs = [slot_hist(a, b, s) for s in range(4)]
        if len(proto_person) < 4 or any(h is None for h in hs):
            team[a:b] = team[a - 1]
            person[a:b] = person[a - 1]
            continue
        P = np.array([proto_person[p] for p in range(4)])
        Hs = np.array(hs)
        cost = np.array([[cv2.compareHist(Hs[s].astype(np.float32), P[p].astype(np.float32),
                                          cv2.HISTCMP_BHATTACHARYYA) for p in range(4)] for s in range(4)])
        # pair-level decision first: far side = team 0 or team 1
        keep = cost[0, 0] + cost[0, 1] + cost[1, 0] + cost[1, 1] + cost[2, 2] + cost[2, 3] + cost[3, 2] + cost[3, 3]
        swap = cost[0, 2] + cost[0, 3] + cost[1, 2] + cost[1, 3] + cost[2, 0] + cost[2, 1] + cost[3, 0] + cost[3, 1]
        far_team = 0 if keep <= swap else 1
        far_people = [0, 1] if far_team == 0 else [2, 3]
        near_people = [2, 3] if far_team == 0 else [0, 1]
        for slots, people in (((0, 1), far_people), ((2, 3), near_people)):
            c = cost[np.ix_(slots, people)]
            straight = c[0, 0] + c[1, 1] <= c[0, 1] + c[1, 0]
            assigned = people if straight else people[::-1]
            for s, p in zip(slots, assigned):
                person[a:b, s] = p
                team[a:b, s] = 0 if p < 2 else 1
    return team, person



def orientation(tracks, segments) -> list[int]:
    """Per segment: the identity (0 / 1) of the pair on the far side."""
    return [int(tracks.team[a, 0]) for a, b in segments]


def main() -> None:
    setup()
    import lightgbm as lgb

    from padel_analyzer.engine import ball as B
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import pipeline
    from padel_analyzer.engine import players as P
    from padel_analyzer.engine import points as PT
    from padel_analyzer.engine.detect import Detections
    from padel_analyzer.eval import pt100

    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"identity": {}, "points": {}}
    rows: dict[str, tuple[np.ndarray, np.ndarray, list]] = {}

    for key in ("F", "M"):
        try:
            name = pt100.MATCHES[key]
            g = pt100.load(key)
            z = np.load(first(f"/kaggle/input/**/cache/{name}_heldout.npz"), allow_pickle=True)
            n = len(z["ball"])
            det = Detections(30.0, (1920, 1080), z["ball"].astype(np.float32), z["pose_box"].astype(np.float32),
                             z["pose_kpt"].astype(np.float32), np.ones(n, bool), np.zeros(n, bool))
            calib = C.Calibration(g.H, (1920, 1080))
            video = first(f"/kaggle/input/**/videos/{name}.mp4")
            log(key, "frames", n, video)
            hists = pipeline.sample_hists(video, det)
            log(key, "hists", len(hists))
            new = P.track(det, calib, hists)
            keep = P._link_identity
            P._link_identity = old_link_identity
            old = P.track(det, calib, hists)
            P._link_identity = keep
            pipeline.sample_hists = lambda *a, **k: hists  # analyze() reuses them
            rep = pipeline.analyze(det, calib, video)
            log(key, "report", rep["match"]["rallies"], "rallies", rep["match"]["hits"], "hits")
            states = json.loads(first(f"/kaggle/input/**/{name}/states.json").read_text())
            tr = transitions(states)
            log(key, "scoreboard transitions", len(tr), "of", len(states), "states")
            shots, rallies = rep["shots"], rep["rallies"]
            gt_serves = [s.frame for s in g.shots if s.kind == "serve"]
            seg = new.segments
            seg_of = lambda f: next((i for i, (a, b) in enumerate(seg) if a <= f < b), len(seg) - 1)  # noqa: E731
            truth_far_row = {}  # segment -> scoreboard row of the pair at the far end
            X, y, meta = [], [], []
            pos = new.person_view().pos_m
            bt = B.track(det.ball)
            for t in tr:
                rs = [r for r in rallies if r["shots"] and t["from"] - 30 <= shots[r["shots"][-1]]["frame"] < t["to"]]
                if len(rs) != 1 or t["server_row"] is None:
                    continue
                r = rs[0]
                first_f = shots[r["shots"][0]]["frame"]
                # where the serve was hit: labelled serve near the rally start, else a detected serve
                sf = next((f for f in gt_serves if first_f - 45 <= f <= first_f + 45), None)
                sp = None
                if sf is not None:
                    pm = g.pos_m[sf]
                    ok = ~np.isnan(pm[:, 0])
                    if ok.any():
                        bx = g.ball[sf]
                        # the server: the labelled player nearest the ball in the image -> his side
                        bb = g.pbox[sf]
                        c = (bb[:, :2] + bb[:, 2:]) / 2
                        dd = np.linalg.norm(c - bx, axis=1) if not np.isnan(bx).any() else np.abs(pm[:, 1] - 10)
                        dd = np.where(ok, dd, np.inf)
                        sp = float(pm[int(np.argmin(dd)), 1])
                if sp is None and shots[r["shots"][0]]["type"] == "serve" and shots[r["shots"][0]].get("pos_m"):
                    sp = shots[r["shots"][0]]["pos_m"][1]
                if sp is None:
                    continue
                server_far = sp < C.NET_Y
                far_row = t["server_row"] if server_far else 1 - t["server_row"]
                truth_far_row[seg_of(first_f)] = far_row
                last = shots[r["shots"][-1]]
                if not last.get("pos_m"):
                    continue
                last_far = last["pos_m"][1] < C.NET_Y
                winner_far = (t["winner_row"] == far_row)
                k = rallies.index(r)
                X.append(PT.rally_features(r, shots, bt.xy, pos, calib, 30.0,
                                           next((b for a, b in seg if b > last["frame"]), n),
                                           shots[rallies[k + 1]["shots"][0]]["frame"] if k + 1 < len(rallies) and rallies[k + 1]["shots"] else None))
                y.append(int(winner_far == last_far))
                meta.append({"frame": last["frame"], "winner_row": t["winner_row"], "far_row": far_row})
            rows[key] = (np.array(X), np.array(y), meta)
            # identity: per labelled segment, does the predicted far pair match the truth (up to one relabelling)?
            for nm, tk in (("old", old), ("new", new)):
                o = orientation(tk, tk.segments)
                pairs = [(o[i], truth_far_row[i]) for i in truth_far_row if i < len(o)]
                agree = sum(a == b for a, b in pairs)
                res["identity"].setdefault(key, {})[nm] = {"segments_checked": len(pairs),
                                                          "correct": max(agree, len(pairs) - agree),
                                                          "segments": len(tk.segments)}
            log(key, "identity", json.dumps(res["identity"][key]), "point rows", len(y), "pos rate", float(np.mean(y)) if y else None)
        except Exception:  # noqa: BLE001
            log("FAILED", key, traceback.format_exc()[-2500:])

    # Kuwait 2025 (unseen, hand-read scoreboard)
    try:
        kd = first("/kaggle/input/**/kuwait_points.json").parent
        rep = json.loads((kd / "report.json").read_text())
        lab = json.loads((kd / "kuwait_points.json").read_text())["points"]
        det = Detections.load(kd / "detections.npz")
        calib = C.Calibration.from_dict(rep["calibration"])
        tk = P.track(det, calib)  # no shirts: slots 0,1 = far pair, so features use sides
        pos, bt, fps, n = tk.person_view().pos_m, B.track(det.ball), det.fps, det.n
        shots, rallies = rep["shots"], rep["rallies"]
        for s in shots:  # team = side here (matches the shirt-free tracks)
            if s.get("pos_m"):
                s["team"] = 0 if s["pos_m"][1] < C.NET_Y else 1
        a_far = lambda t: 75 <= t < 299  # Coello/Tapia's end, from serve positions + changes of ends  # noqa: E731
        X, y, meta, prev = [], [], [], 0
        for p in lab:
            rs = [r for r in rallies if r["shots"] and prev < shots[r["shots"][-1]]["t"] <= p["by_s"]]
            prev = p["by_s"]
            if not rs:
                continue
            r = rs[-1]
            last = shots[r["shots"][-1]]
            if not last.get("pos_m"):
                continue
            winner_far = a_far(last["t"]) if p["winner_team"] == 0 else not a_far(last["t"])
            k = rallies.index(r)
            X.append(PT.rally_features(r, shots, bt.xy, pos, calib, fps, n,
                                       shots[rallies[k + 1]["shots"][0]]["frame"] if k + 1 < len(rallies) and rallies[k + 1]["shots"] else None))
            y.append(int(winner_far == (last["pos_m"][1] < C.NET_Y)))
            meta.append({"t": last["t"]})
        rows["K"] = (np.array(X), np.array(y), meta)
        log("K point rows", len(y), "pos rate", float(np.mean(y)))
    except Exception:  # noqa: BLE001
        log("FAILED kuwait", traceback.format_exc()[-2500:])

    params = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=8, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
    keys = [k for k in rows if len(rows[k][1])]
    res["points"]["rows"] = {k: int(len(rows[k][1])) for k in keys}
    res["points"]["last_hitter_won_rate"] = {k: round(float(np.mean(rows[k][1])), 3) for k in keys}
    for te in keys:
        tr = [k for k in keys if k != te]
        if not tr:
            continue
        Xtr = np.vstack([rows[k][0] for k in tr])
        ytr = np.concatenate([rows[k][1] for k in tr])
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=150)
        p = bst.predict(rows[te][0])
        yt = rows[te][1]
        acc = float(((p >= 0.5) == yt).mean())
        maj = float(max(yt.mean(), 1 - yt.mean()))
        base = float(max(ytr.mean(), 1 - ytr.mean()))
        res["points"][f"test_{te}"] = {"n": int(len(yt)), "accuracy": round(acc, 3),
                                        "always_majority_of_train": round(float(((ytr.mean() >= 0.5) == yt).mean()), 3),
                                        "test_majority_rate": round(maj, 3)}
        log("points LOMO", te, json.dumps(res["points"][f"test_{te}"]))
    if keys:
        Xa = np.vstack([rows[k][0] for k in keys])
        ya = np.concatenate([rows[k][1] for k in keys])
        bst = lgb.train(params, lgb.Dataset(Xa, ya, feature_name=PT.FEATURES), num_boost_round=150)
        (OUT / "models").mkdir(exist_ok=True)
        bst.save_model(str(OUT / "models" / "point_lgbm.txt"))
        (OUT / "models" / "point_lgbm.json").write_text(json.dumps({"rows": res["points"]["rows"]}))
        imp = dict(zip(PT.FEATURES, bst.feature_importance("gain").round(1).tolist()))
        res["points"]["importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])
        np.savez(OUT / "point_rows.npz", **{f"X_{k}": rows[k][0] for k in keys}, **{f"y_{k}": rows[k][1] for k in keys})
    (OUT / "points.json").write_text(json.dumps(res, indent=1, default=str))
    log("final", json.dumps(res, default=str)[:3000])


if __name__ == "__main__":
    main()
