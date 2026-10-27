"""Padel kernel: robust audio+video hit fusion (modality dropout) + padel shot-type validation on PT100.

Reuses padel-audio outputs (per-rally detections, out-of-fold audio probabilities) — no re-detection.

  1. Hit GBM variants on Decorte (11 tournaments, tournament-grouped folds):
       fused            as in padel-audio
       fused_dropout    + training copies with the ball features blanked and with the audio blanked
     evaluated (a) as recorded and (b) with every ball feature blanked ("no ball", a club camera's worst case).
     Production model = fused_dropout on all Decorte rallies + PadelTracker100 held-out rows.
  2. Padel shot-type detector (padel-venues shot_types) on PadelTracker100 GT hits: classes mapped onto
     serve / forehand / backhand / smash / other -> accuracy + coverage.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
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
    cache = first("/kaggle/input/**/out/cache/2022_BCN_FinalF_1_heldout.npz").parent
    (REPO / "data" / "kaggle_out" / "gpu" / "out").mkdir(parents=True)
    os.symlink(cache, REPO / "data" / "kaggle_out" / "gpu" / "out" / "cache")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "kaggle" / "audio"))


BALL_COLS = ["obs_frac", "has_ball", "cos_turn3", "cos_turn6", "speed_b3", "speed_a3", "speed_ratio3", "acc3",
             "acc6", "vy_flip", "d_wrist", "d_wrist_m3", "d_wrist_p3", "d_wrist_2nd", "in_box", "rel_x", "rel_y",
             "acc3_m2", "acc3_p2"]


def stage_fusion() -> dict:
    import lightgbm as lgb

    import padel_audio as PA
    import train_events as T
    from padel_analyzer.engine import audio as A
    from padel_analyzer.engine import ball as B
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import events as E
    from padel_analyzer.engine import players as P
    from padel_analyzer.engine.detect import Detections
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
        tracks = P.track(det, calib)
        bt = B.track_for_events(det.ball)
        aud = A.video_features(oof[f"t__{n}"], oof[f"p__{n}"], det.n, det.fps, det.step, float(oof[f"thr__{n}"]))
        X, nearest = E.features(bt, tracks, aud)
        gt = [int(round(a * det.fps)) for a, _ in hits[n]]
        data[n] = {"X": X, "nearest": nearest, "y": E.labels(det.n, gt, radius=2), "gt": gt}
    ball_idx = [E.FEATURES.index(c) for c in BALL_COLS]
    aud_idx = [E.FEATURES.index(c) for c in E.AUDIO_FEATURES]

    def blank(X, idx):
        X = X.copy()
        X[:, idx] = np.nan
        return X

    def train_rows(train_names, dropout):
        X = np.vstack([data[n]["X"] for n in train_names])
        y = np.concatenate([data[n]["y"] for n in train_names])
        if dropout:
            X, y = np.vstack([X, blank(X, ball_idx), blank(X, aud_idx)]), np.concatenate([y, y, y])
        return X, y

    def fit(train_names, dropout):
        X, y = train_rows(train_names, dropout)
        bst = lgb.train(T.HIT_PARAMS, lgb.Dataset(X, y), num_boost_round=400)
        best = (0.0, 0.5)
        for thr in np.arange(0.2, 0.9, 0.05):
            tp = npred = ngt = 0
            for n in train_names:
                hs = E.pick_peaks(bst.predict(data[n]["X"]), data[n]["nearest"], thr, nms=8)
                a, b, c = PA.event_f1([h.frame for h in hs], data[n]["gt"], 5)
                tp, npred, ngt = tp + a, npred + b, ngt + c
            best = max(best, (PA.prf(tp, npred, ngt)["f1"], float(thr)))
        return bst, best[1]

    res = {}
    for variant, dropout in (("fused", False), ("fused_dropout", True)):
        out = {}
        for cond, cols in (("as_recorded", None), ("no_ball", ball_idx), ("no_audio", aud_idx)):
            tot = [0, 0, 0]
            for test_t in fold_sets:
                tr = [n for n in names if PA.tournament(n) not in test_t]
                te = [n for n in names if PA.tournament(n) in test_t]
                bst, thr = fit(tr, dropout)
                for n in te:
                    X = data[n]["X"] if cols is None else blank(data[n]["X"], cols)
                    hs = E.pick_peaks(bst.predict(X), data[n]["nearest"], thr, nms=8)
                    a, b, c = PA.event_f1([h.frame for h in hs], data[n]["gt"], 5)
                    tot = [tot[0] + a, tot[1] + b, tot[2] + c]
            out[cond] = PA.prf(*tot)
        res[variant] = out
        log("fusion", variant, json.dumps(out))

    # production: fused + modality dropout on every Decorte rally + PT100 held-out (video-only) rows
    X, y = train_rows(names, True)
    for k in "FM":
        d = T.prepare(k, "heldout")
        X = np.vstack([X, d["X"]])
        y = np.concatenate([y, E.labels(d["n"], d["gt_hits"], radius=2)])
    bst = lgb.train(T.HIT_PARAMS, lgb.Dataset(X, y, feature_name=E.FEATURES), num_boost_round=400)
    _, thr = fit(names, True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    bst.save_model(str(OUT / "models" / "hit_lgbm.txt"))
    (OUT / "models" / "hit_lgbm.json").write_text(json.dumps({"threshold": thr, "source": "decorte+pt100 fused, modality dropout"}))
    res["production_threshold"] = thr
    return res


MAP = {"Golpe de derecha": "forehand", "Volea de derecha": "forehand", "Golpe de reves": "backhand",
       "Volea de reves": "backhand", "Remate": "smash", "Bandeja": "smash", "Saque": "serve",
       "Globo": "other", "Salida de pared": "other", "Contrapared": "other"}


def stage_shot_types() -> dict:
    import cv2
    from ultralytics import YOLO

    from padel_analyzer.eval import pt100

    w = first("/kaggle/input/**/models/shot_types/best.pt")
    model = YOLO(str(w))
    names = model.names
    vids = first("/kaggle/input/**/pt100/videos/2022_BCN_FinalF_1.mp4").parent
    res = {}
    for k in "FM":
        g = pt100.load(k)
        cap = cv2.VideoCapture(str(vids / f"{pt100.MATCHES[k]}.mp4"))
        n_ok = n_cov = 0
        conf: dict[str, dict[str, int]] = {}
        shots = [s for s in g.shots if s.frame < g.shot_range[1]]
        for s in shots:
            f = s.frame
            b = g.ball[f]
            boxes = g.pbox[f]
            if np.isnan(b).any():
                continue
            d = [np.inf if np.isnan(bx[0]) else np.linalg.norm(g.pkpt[f, i, [9, 10], :2] - b, axis=1).min()
                 for i, bx in enumerate(boxes)]
            hb = boxes[int(np.argmin(d))]
            votes: dict[str, float] = {}
            for o in (-3, -1, 0, 1, 3):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f + o)
                ok, img = cap.read()
                if not ok:
                    continue
                r = model.predict(img, imgsz=640, conf=0.25, verbose=False, device=0)[0]
                for bb, c, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    lab = MAP.get(names[int(c)])
                    if lab is None:
                        continue
                    x0, y0 = max(bb[0], hb[0]), max(bb[1], hb[1])
                    x1, y1 = min(bb[2], hb[2]), min(bb[3], hb[3])
                    inter = max(0, x1 - x0) * max(0, y1 - y0)
                    ua = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (hb[2] - hb[0]) * (hb[3] - hb[1]) - inter
                    if ua > 0 and inter / ua > 0.2:
                        votes[lab] = votes.get(lab, 0) + float(sc)
            if votes:
                n_cov += 1
                pred = max(votes, key=votes.get)
                n_ok += pred == s.kind
                conf.setdefault(s.kind, {}).setdefault(pred, 0)
                conf[s.kind][pred] += 1
        res[k] = {"shots": len(shots), "coverage": round(n_cov / max(1, len(shots)), 4),
                  "accuracy_when_covered": round(n_ok / max(1, n_cov), 4), "confusion": conf}
        log("shot types", k, json.dumps(res[k])[:400])
    return res


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "lightgbm"], check=False)
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    res = {}
    for name, fn in (("fusion", stage_fusion), ("shot_types_on_pt100", stage_shot_types)):
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001
            res[name] = "FAILED " + traceback.format_exc()[-2000:]
        (OUT / "fusion.json").write_text(json.dumps(res, indent=1, default=str))
    log("final", json.dumps(res, default=str)[:3000])


if __name__ == "__main__":
    main()
