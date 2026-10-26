"""Train + cross-match evaluate the hit detector and stroke classifier on PadelTracker100.

  python scripts/train_events.py --source oracle      # GT ball + GT pose (upper bound)
  python scripts/train_events.py --source heldout     # Kaggle held-out detections (the real number)
  python scripts/train_events.py --source heldout --save   # + write models/hit_lgbm.txt, shot_lgbm.txt

Cross-match: fit on F -> score on M, fit on M -> score on F. --save refits on both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from padel_analyzer.engine import ball as B  # noqa: E402
from padel_analyzer.engine import court as C  # noqa: E402
from padel_analyzer.engine import events as E  # noqa: E402
from padel_analyzer.engine import players as P  # noqa: E402
from padel_analyzer.engine import shots as S  # noqa: E402
from padel_analyzer.engine.detect import Detections  # noqa: E402
from padel_analyzer.eval import metrics as M  # noqa: E402
from padel_analyzer.eval import pt100  # noqa: E402

CACHE_DIR = ROOT / "data" / "kaggle_out" / "gpu" / "out" / "cache"
RGB_HELDOUT = ROOT / "data" / "kaggle_out" / "stroke" / "out" / "stroke_rgb_heldout.npz"
TOL = 7


def oracle_dets(g: pt100.MatchGT, n: int) -> Detections:
    pb = np.zeros((n, 10, 5), np.float32)
    pk = np.zeros((n, 10, 17, 3), np.float32)
    pb[:, :5, :4] = np.nan_to_num(g.pbox[:n])
    pb[:, :5, 4] = np.where(np.isnan(g.pbox[:n, :, 0]), 0, 1)
    pk[:, :5] = g.pkpt[:n]
    pk[:, :5, :, 2] = np.where(pk[:, :5, :, 2] > 0, 1.0, 0)
    ball = np.zeros((n, 5, 5), np.float32)
    ok = ~np.isnan(g.ball[:n, 0])
    ball[ok, 0, :2] = g.ball[:n][ok] - 4
    ball[ok, 0, 2:4] = g.ball[:n][ok] + 4
    ball[ok, 0, 4] = 1
    return Detections(30.0, (1920, 1080), ball, pb, pk, np.ones(n, bool), np.zeros(n, bool))


def heldout_dets(g: pt100.MatchGT, n: int) -> Detections:
    z = np.load(CACHE_DIR / f"{pt100.MATCHES[g.key]}_heldout.npz", allow_pickle=True)
    return Detections(30.0, (1920, 1080), z["ball"][:n].astype(np.float32), z["pose_box"][:n].astype(np.float32),
                      z["pose_kpt"][:n].astype(np.float32), np.ones(n, bool), np.zeros(n, bool))


def prepare(key: str, source: str) -> dict:
    g = pt100.load(key)
    n = g.shot_range[1]
    det = oracle_dets(g, n) if source == "oracle" else heldout_dets(g, n)
    calib = C.Calibration(g.H, (1920, 1080))
    tracks = P.track(det, calib)
    bt = B.track_for_events(det.ball)
    X, nearest = E.features(bt, tracks)
    gt_hits = [s.frame for s in g.shots if s.frame < n]
    y = E.labels(n, gt_hits)
    return {"g": g, "n": n, "tracks": tracks, "ball": bt, "X": X, "nearest": nearest, "y": y, "gt_hits": gt_hits,
            "ball_pr": M.ball_pr(bt.xy, g.ball[:n])}


HIT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, scale_pos_weight=3.0)
SHOT_PARAMS = dict(objective="multiclass", num_class=len(S.CLASSES), learning_rate=0.05, num_leaves=15,
                   min_data_in_leaf=8, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def fit_hit(ds: list[dict]) -> tuple[lgb.Booster, float]:
    X = np.vstack([d["X"] for d in ds])
    y = np.concatenate([d["y"] for d in ds])
    bst = lgb.train(HIT_PARAMS, lgb.Dataset(X, y, feature_name=E.FEATURES), num_boost_round=400)
    # threshold maximising in-sample event F1 (on the training matches only)
    best = (0.0, 0.5)
    for thr in np.arange(0.2, 0.85, 0.05):
        f1s = []
        for d in ds:
            hits = E.pick_peaks(bst.predict(d["X"]), d["nearest"], thr)
            f1s.append(M.event_f1([h.frame for h in hits], d["gt_hits"], TOL)["f1"])
        best = max(best, (float(np.mean(f1s)), float(thr)))
    return bst, best[1]


def shot_xy(d: dict) -> tuple[np.ndarray, np.ndarray, list[E.Hit]]:
    g = d["g"]
    shots = [s for s in g.shots if s.frame < d["n"]]
    hits = [E.Hit(s.frame, int(_hitter(d, s.frame)), 1.0) for s in shots]
    X = S.features(hits, d["ball"], d["tracks"], 30.0)
    y = np.array([S.CLASSES.index(s.kind) for s in shots])
    return X, y, hits


def _hitter(d: dict, f: int) -> int:
    """Slot closest to the ball around frame f (GT-hit evaluation of the stroke model)."""
    X = d["X"]
    c = E.FEATURES.index("d_wrist")
    lo, hi = max(0, f - 3), min(len(X), f + 4)
    w = X[lo:hi, c]
    if np.isnan(w).all():
        return int(d["nearest"][f])
    return int(d["nearest"][lo + int(np.nanargmin(w))])


def fit_shot(ds: list[dict]) -> lgb.Booster:
    X = np.vstack([shot_xy(d)[0] for d in ds])
    y = np.concatenate([shot_xy(d)[1] for d in ds])
    return lgb.train(SHOT_PARAMS, lgb.Dataset(X, y, feature_name=S.feature_names()), num_boost_round=300)


def rgb_fusion(te: str, d: dict, P_pose: np.ndarray, y: np.ndarray) -> dict | None:
    """Accuracy of w * RGB-clip + (1 - w) * pose/context probabilities on the same GT shots."""
    if not RGB_HELDOUT.exists():
        return None
    z = np.load(RGB_HELDOUT)
    if f"P_{te}" not in z.files:
        return None
    rgb = dict(zip(z[f"frames_{te}"].tolist(), z[f"P_{te}"]))
    frames = [s.frame for s in d["g"].shots if s.frame < d["n"]]
    keep = [i for i, f in enumerate(frames) if f in rgb]
    R = np.array([rgb[frames[i]] for i in keep])
    Pp, yy = P_pose[keep], y[keep]
    out = {"n": len(keep)}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        out[f"w{w}"] = round(float(((w * R + (1 - w) * Pp).argmax(1) == yy).mean()), 4)
    return out


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for c in range(len(S.CLASSES)):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        if tp + fn == 0:
            continue
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["oracle", "heldout"], default="oracle")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    data = {k: prepare(k, args.source) for k in "FM"}
    report: dict = {"source": args.source}
    for k, d in data.items():
        report[f"ball_{k}"] = d["ball_pr"]
    for tr, te in (("F", "M"), ("M", "F")):
        bst, thr = fit_hit([data[tr]])
        d = data[te]
        hits = E.pick_peaks(bst.predict(d["X"]), d["nearest"], thr)
        ev = M.event_f1([h.frame for h in hits], d["gt_hits"], TOL)
        heur = E.pick_peaks(E.heuristic_score(d["X"]), d["nearest"], 0.15)
        evh = M.event_f1([h.frame for h in heur], d["gt_hits"], TOL)
        sb = fit_shot([data[tr]])
        Xs, ys, _ = shot_xy(d)
        Ps = sb.predict(Xs)
        ps = Ps.argmax(1)
        fusion = rgb_fusion(te, d, Ps, ys)
        # end-to-end: stroke type of predicted hits matched to GT hits
        gt_kind = {s.frame: S.CLASSES.index(s.kind) for s in d["g"].shots}
        pred_by_frame = {h.frame: h for h in hits}
        matched = [(pred_by_frame[p], gt_kind[g]) for p, g in ev["pairs"]]
        e2e = None
        if matched:
            Xm = S.features([m[0] for m in matched], d["ball"], d["tracks"], 30.0)
            e2e = float((sb.predict(Xm).argmax(1) == np.array([m[1] for m in matched])).mean())
        report[f"test_{te}"] = {
            "hit": {k: v for k, v in ev.items() if k != "pairs"}, "hit_threshold": thr,
            "hit_heuristic": {k: v for k, v in evh.items() if k != "pairs"},
            "stroke_acc_on_gt_hits": round(float((ps == ys).mean()), 4),
            "stroke_fusion_with_rgb": fusion,
            "stroke_macro_f1": round(macro_f1(ys, ps), 4),
            "stroke_acc_end_to_end": round(e2e, 4) if e2e is not None else None,
            "confusion": _confusion(ys, ps),
        }
    print(json.dumps(report, indent=1))
    out = ROOT / "data" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"events_{args.source}.json").write_text(json.dumps(report, indent=1))
    if args.save:
        mdir = ROOT / "models"
        mdir.mkdir(exist_ok=True)
        bst, thr = fit_hit(list(data.values()))
        bst.save_model(str(mdir / "hit_lgbm.txt"))
        (mdir / "hit_lgbm.json").write_text(json.dumps({"threshold": thr, "source": args.source}))
        fit_shot(list(data.values())).save_model(str(mdir / "shot_lgbm.txt"))
        print("saved models to", mdir)


def _confusion(y: np.ndarray, p: np.ndarray) -> dict:
    return {S.CLASSES[c]: {S.CLASSES[q]: int(((y == c) & (p == q)).sum()) for q in range(len(S.CLASSES))}
            for c in range(len(S.CLASSES))}


if __name__ == "__main__":
    main()
