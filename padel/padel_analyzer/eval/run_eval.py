"""Cross-match evaluation on PadelTracker100 held-out detections -> data/eval/heldout.json (+ markdown rows).

  python -m padel_analyzer.eval.run_eval [--frames 20100]

Needs the GPU kernel caches (data/kaggle_out/gpu/out/cache/*_heldout.npz) and, for hits/strokes,
models trained cross-match by scripts/train_events.py (it is called here).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..engine import ball as B
from ..engine import court as C
from ..engine import players as P
from ..engine import rallies as R
from ..engine.detect import Detections
from ..engine.events import Hit
from . import metrics as M
from . import pt100

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "kaggle_out" / "gpu" / "out" / "cache"


def _boxes(b: np.ndarray, min_conf: float) -> np.ndarray:
    out = b[..., :4].copy()
    out[b[..., 4] < min_conf] = np.nan
    return out


def eval_match(key: str, n_max: int) -> dict:
    g = pt100.load(key)
    z = np.load(CACHE / f"{pt100.MATCHES[key]}_heldout.npz", allow_pickle=True)
    n = min(n_max or g.n, g.n, len(z["ball"]))
    det = Detections(30.0, (1920, 1080), z["ball"][:n].astype(np.float32), z["pose_box"][:n].astype(np.float32),
                     z["pose_kpt"][:n].astype(np.float32), np.ones(n, bool), np.zeros(n, bool))
    calib = C.Calibration(g.H, (1920, 1080))
    out: dict = {"frames": n}

    # ball: raw argmax vs trajectory linker
    cand = det.ball
    best = cand[np.arange(n), cand[:, :, 4].argmax(1)]
    raw = np.where(best[:, 4:5] >= 0.25, np.c_[(best[:, 0] + best[:, 2]) / 2, (best[:, 1] + best[:, 3]) / 2], np.nan)
    bt = B.track(cand)
    out["ball_argmax"] = M.ball_pr(raw, g.ball[:n])
    out["ball_linked"] = M.ball_pr(bt.xy, g.ball[:n])
    out["ball_linked_observed_only"] = M.ball_pr(np.where(bt.observed[:, None], bt.xy, np.nan), g.ball[:n])

    # players: detection (fine-tuned every frame; stock yolo11m-pose on every 10th frame)
    every10 = np.arange(0, n, 10)
    out["players_det_finetuned"] = M.player_det_pr(_boxes(det.pose_box, 0.3), g.pbox[:n], frames=every10)
    out["players_det_stock"] = M.player_det_pr(_boxes(z["base_box"][:n].astype(np.float32), 0.3), g.pbox[:n],
                                               frames=every10)
    tracks = P.track(det, calib)
    out["position"] = M.position_error(tracks.pos_m, g.pos_m[:n], frames=every10)
    gt = P._smooth(pt100.link_tracks(g.pos_m[:n]), 30.0, 30)
    gd = pt100.path_length(gt).reshape(2, 2).sum(1)
    od = pt100.path_length(tracks.pos_m).reshape(2, 2).sum(1)
    out["pair_distance_m"] = {"gt": gd.round(1).tolist(), "ours": od.round(1).tolist(),
                              "err_pct": (100 * (od - gd) / gd).round(2).tolist()}
    out["segments"] = len(tracks.segments)
    out["segments_gt_tracks"] = len(P.track(T_oracle(g, n), calib).segments)

    # rallies: GT hits grouped with the same rules (predicted count is added in main, needs the other match)
    gt_hits = [Hit(s.frame, 0, 1.0) for s in g.shots if s.frame < min(n, g.shot_range[1])]
    gt_rallies = R.group(gt_hits, 30.0, n, tracks.segments)
    out["rallies_gt"] = len(gt_rallies)

    # ball inside rallies only: PT100 leaves the ball unlabelled between points even when it is visible
    in_rally = np.zeros(n, bool)
    for r in gt_rallies:
        in_rally[max(0, r.start):min(n, r.end + 1)] = True
    lab = np.zeros(n, bool)
    lab[: min(n, g.shot_range[1])] = True
    fr = np.where(in_rally & lab)[0]
    out["ball_in_rally_frames"] = int(len(fr))
    out["ball_argmax_in_rally"] = M.ball_pr(raw, g.ball[:n], frames=fr)
    out["ball_linked_in_rally"] = M.ball_pr(bt.xy, g.ball[:n], frames=fr)
    grid = {}
    for mc in (0.08, 0.15, 0.25, 0.35, 0.5):
        for mf in (0, 4, 8):
            b = B.track(cand, min_conf=mc, max_fill=mf)
            grid[f"c{mc}_f{mf}"] = {"all": M.ball_pr(b.xy, g.ball[:n]), "in_rally": M.ball_pr(b.xy, g.ball[:n], frames=fr)}
    out["ball_grid"] = grid
    out["_cache"] = {"det": det, "calib": calib, "tracks": tracks, "g": g}
    return out


def T_oracle(g, n):
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_events as T

    return T.oracle_dets(g, n)


def predicted_rallies(train: dict, test: dict) -> dict:
    """Hit model fit on the other match -> predicted hits on this one -> rallies."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_events as T

    tr, te = T.prepare(train["key"], "heldout"), T.prepare(test["key"], "heldout")
    bst, thr = T.fit_hit([tr])
    from ..engine.events import pick_peaks

    hits = pick_peaks(bst.predict(te["X"]), te["nearest"], thr)
    seg = test["_cache"]["tracks"].segments
    pred = R.group(hits, 30.0, te["n"], seg)
    grid = {f"gap{gap}_{'seg' if use_seg else 'noseg'}": len(R.group(hits, 30.0, te["n"], seg if use_seg else None,
                                                                    max_gap_s=gap))
            for gap in (3.0, 3.5, 4.0, 4.5, 5.0) for use_seg in (True, False)}
    return {"rallies_pred": len(pred), "rally_shots_pred_mean": round(float(np.mean([len(r.hits) for r in pred])), 2)
            if pred else None, "rallies_pred_grid": grid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="limit frames per match (0 = all)")
    args = ap.parse_args()
    res = {k: eval_match(k, args.frames) for k in "FM"}
    for k in "FM":
        res[k]["key"] = k
    for tr, te in (("M", "F"), ("F", "M")):
        res[te].update(predicted_rallies(res[tr], res[te]))
    # ball linker: choose the setting on one match (in-rally F1), report it on the other
    for tune, test in (("F", "M"), ("M", "F")):
        g = res[tune]["ball_grid"]
        best = max(g, key=lambda k: g[k]["in_rally"]["f1"])
        res[test]["ball_linker_tuned_on_other"] = {"setting": best, **res[test]["ball_grid"][best]}
    for k in "FM":
        res[k].pop("_cache", None)
        res[k].pop("key", None)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_events.py"), "--source", "heldout"], check=True,
                   stdout=subprocess.DEVNULL)
    ev = json.loads((ROOT / "data" / "eval" / "events_heldout.json").read_text())
    for k in "FM":
        res[k]["events"] = ev[f"test_{k}"]
    out = ROOT / "data" / "eval" / "heldout.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
