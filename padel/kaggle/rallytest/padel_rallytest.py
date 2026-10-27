"""CPU kernel: rally segmentation on the Decorte rallies (each clip is exactly one point).

Hits are predicted out-of-fold (tournament-grouped, fused audio+video GBM with modality dropout, as in
padel-fusion) on the padel-audio detections, then grouped with engine.rallies. Truth = 1 rally per clip.
Reports, for ground-truth hits and for predicted hits, with and without the dead-ball rule:
rallies per clip, clips split (> 1), clips lost (0 left).
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working/out")
REPO = Path("/tmp/padel")


def first(p: str) -> Path | None:
    h = sorted(glob.glob(p, recursive=True))
    return Path(h[0]) if h else None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    code = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    shutil.copytree(code, REPO)
    labels = first("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json").parent
    (REPO / "data" / "pt100").mkdir(parents=True)
    shutil.copytree(labels, REPO / "data" / "pt100" / "labels")
    for p in (REPO, REPO / "scripts", REPO / "kaggle" / "audio"):
        sys.path.insert(0, str(p))


def main() -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=False)
    setup()
    import lightgbm as lgb

    import padel_audio as PA
    import train_events as T
    from padel_analyzer.engine import audio as A
    from padel_analyzer.engine import ball as B
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import events as E
    from padel_analyzer.engine import players as P
    from padel_analyzer.engine import rallies as R
    from padel_analyzer.engine.detect import Detections
    from padel_analyzer.engine.events import Hit
    from padel_analyzer.eval import pt100

    dec = first("/kaggle/input/**/CVSPORTS_Padel/metadata/hits.csv").parents[1]
    hits = PA.load_hits(dec)
    names = sorted(hits)
    fold_sets = PA.folds(names)
    det_dir = first("/kaggle/input/**/out/det/20230528_VIGO_00.mp4.npz").parent
    oof = np.load(first("/kaggle/input/**/out/audio_oof.npz"))
    calib = C.Calibration(pt100.load("F").H, (1920, 1080))
    data = {}
    for n in names:
        det = Detections.load(det_dir / (n + ".npz"))
        st = P.track(det, calib)
        bt = B.track_for_events(det.ball)
        aud = A.video_features(oof[f"t__{n}"], oof[f"p__{n}"], det.n, det.fps, det.step, float(oof[f"thr__{n}"]))
        X, nearest = E.features(bt, st, aud)
        gt = [int(round(a * det.fps)) for a, _ in hits[n]]
        data[n] = {"X": X, "nearest": nearest, "y": E.labels(det.n, gt, radius=2), "gt": gt, "st": st, "det": det}
    log("features", len(data))
    # same modality dropout as the production model (padel-fusion): copies without ball and without audio
    ball_cols = ["obs_frac", "has_ball", "cos_turn3", "cos_turn6", "speed_b3", "speed_a3", "speed_ratio3", "acc3",
                 "acc6", "vy_flip", "d_wrist", "d_wrist_m3", "d_wrist_p3", "d_wrist_2nd", "in_box", "rel_x",
                 "rel_y", "acc3_m2", "acc3_p2"]
    drop = [[E.FEATURES.index(c) for c in ball_cols], [E.FEATURES.index(c) for c in E.AUDIO_FEATURES]]

    def train(tr):
        X = np.vstack([data[n]["X"] for n in tr])
        y = np.concatenate([data[n]["y"] for n in tr])
        copies = [X]
        for idx in drop:
            Xb = X.copy()
            Xb[:, idx] = np.nan
            copies.append(Xb)
        bst = lgb.train(T.HIT_PARAMS, lgb.Dataset(np.vstack(copies), np.tile(y, 3)), num_boost_round=400)
        best = (0.0, 0.5)
        for thr in np.arange(0.2, 0.9, 0.05):
            tot = [0, 0, 0]
            for n in tr:
                hs = E.pick_peaks(bst.predict(data[n]["X"]), data[n]["nearest"], thr, nms=8)
                tot = [a + b for a, b in zip(tot, PA.event_f1([h.frame for h in hs], data[n]["gt"], 5))]
            best = max(best, (PA.prf(*tot)["f1"], float(thr)))
        return bst, best[1]

    pred = {}
    for test_t in fold_sets:
        tr = [n for n in names if PA.tournament(n) not in test_t]
        bst, thr = train(tr)
        for n in names:
            if PA.tournament(n) in test_t:
                pred[n] = E.pick_peaks(bst.predict(data[n]["X"]), data[n]["nearest"], thr, nms=8)
    log("predicted", sum(len(v) for v in pred.values()), "hits vs gt", sum(len(d["gt"]) for d in data.values()))

    def rallies_for(n, hs, dead):
        d, st = data[n]["det"], data[n]["st"]
        rs = R.group(hs, d.fps, d.n, st.segments)
        if dead:
            team = [0 if st.person[h.frame, h.slot] < 2 else 1 for h in hs]
            live = sorted(i for r in rs if not R.dead_ball(r, team) for i in r.hits)
            hs = [hs[i] for i in live]
            rs = R.group(hs, d.fps, d.n, st.segments) if hs else []
        return rs

    res, per_clip = {}, {}
    for src in ("gt_hits", "pred_hits"):
        for dead in (False, True):
            counts = []
            for n in names:
                if src == "gt_hits":
                    hs = [Hit(f, int(data[n]["nearest"][min(f, len(data[n]["nearest"]) - 1)]), 1.0) for f in data[n]["gt"]]
                else:
                    hs = pred[n]
                c = len(rallies_for(n, hs, dead))
                counts.append(c)
                per_clip.setdefault(n, {})[f"{src}{'_dead' if dead else ''}"] = c
            k = f"{src}{' + dead_ball' if dead else ''}"
            res[k] = {"clips": len(counts), "rallies": int(sum(counts)), "exact_1": sum(c == 1 for c in counts),
                      "split": sum(c > 1 for c in counts), "lost": sum(c == 0 for c in counts),
                      "hist": {str(v): counts.count(v) for v in sorted(set(counts))}}
            log(k, json.dumps(res[k]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rallytest.json").write_text(json.dumps({"summary": res, "per_clip": per_clip}, indent=1))


if __name__ == "__main__":
    main()
