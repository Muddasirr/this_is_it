"""Padel GPU kernel: multimodal (audio + video) hit detection on the Decorte et al. CVPRW 2024 dataset.

Data: 99 rallies, 11 WPT 2023 tournaments (indoor + outdoor, blue / grey / purple courts), 25 fps, audio,
2,377 hit onset/offset annotations (padel-decorte-prep output). Folds are grouped by TOURNAMENT, so every
reported number is on venues the models never saw (stricter than the paper's random rally splits).

  A. audio CRNN (engine/audio.py; paper architecture + focal loss), 4 tournament folds -> out-of-fold
     per-frame probabilities, event F1 at ±250 ms (paper's collar) and ±100 ms; final model on all rallies.
  B. video: ball (ball_v2) + pose (pose_all) detections on every rally (2 GPUs).
  C. hit classifier (GBM, engine/events.py features) with the same folds: video-only vs audio-only vs fused.
  D. production fused hit model: all Decorte rallies + PadelTracker100 held-out rows (video-only, audio NaN).
Outputs /kaggle/working/out/{audio.json, models/audio_hit.pt, models/hit_lgbm.*, logs}.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working/out")
REPO = Path("/tmp/padel")
FPS = 25.0
TOL_FRAMES = {"200ms": 5, "280ms": 7}
COLLARS = (0.25, 0.1)


def first(p: str) -> Path | None:
    h = sorted(glob.glob(p, recursive=True))
    return Path(h[0]) if h else None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup() -> tuple[Path, Path]:
    code = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(code, REPO)
    labels = first("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json").parent
    (REPO / "data" / "pt100").mkdir(parents=True)
    shutil.copytree(labels, REPO / "data" / "pt100" / "labels")
    cache = first("/kaggle/input/**/out/cache/2022_BCN_FinalF_1_heldout.npz").parent
    (REPO / "data" / "kaggle_out" / "gpu" / "out").mkdir(parents=True)
    os.symlink(cache, REPO / "data" / "kaggle_out" / "gpu" / "out" / "cache")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    dec = first("/kaggle/input/**/CVSPORTS_Padel/metadata/hits.csv").parents[1]
    return REPO, dec


def load_hits(dec: Path) -> dict[str, list[tuple[float, float]]]:
    hits: dict[str, list[tuple[float, float]]] = {}
    with open(dec / "metadata" / "hits.csv") as f:
        for r in csv.DictReader(f):
            hits.setdefault(r["filename"], []).append((float(r["start"]), float(r["end"])))
    return {k: sorted(v) for k, v in hits.items()}


def tournament(name: str) -> str:
    return "_".join(name.split("_")[:2])


def folds(names: list[str], k: int = 4) -> list[set[str]]:
    """Tournaments assigned to k folds, balancing hit counts (largest first)."""
    tours: dict[str, int] = {}
    for n in names:
        tours[tournament(n)] = tours.get(tournament(n), 0) + 1
    bins: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for t in sorted(tours, key=lambda t: -tours[t]):
        i = int(np.argmin(load))
        bins[i].append(t)
        load[i] += tours[t]
    return [set(b) for b in bins]


def event_f1(pred: list[float], gt: list[float], tol: float) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in sorted(pred):
        best, bd = None, tol + 1e-9
        for i, g in enumerate(gt):
            if i not in used and abs(g - p) <= bd:
                best, bd = i, abs(g - p)
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(pred), len(gt)


def prf(tp, npred, ngt) -> dict:
    p, r = tp / max(1, npred), tp / max(1, ngt)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4),
            "tp": tp, "pred": npred, "gt": ngt}


# ---------------------------------------------------------------- A. audio
def stage_audio(dec: Path, hits: dict, fold_sets: list[set[str]]) -> dict:
    import torch

    from padel_analyzer.engine import audio as A

    names = sorted(hits)
    feats, targets = {}, {}
    for n in names:
        y = A.load_audio(dec / "rallies" / n)
        m = A.logmel(y)
        t = A.frame_times(len(m))
        tgt = np.zeros(len(m), np.float32)
        for a, b in hits[n]:
            tgt[(t >= a - 0.02) & (t <= b + 0.02)] = 1
        feats[n], targets[n] = m, tgt
    log("audio features", len(feats), "rallies")

    def train(train_names: list[str], epochs: int = 120, seed: int = 0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        allf = np.concatenate([feats[n] for n in train_names])
        mean, std = allf.mean(0), allf.std(0) + 1e-6
        model = A.build_model().cuda()
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        L = A.SEQ
        for ep in range(epochs):
            model.train()
            xs, ys = [], []
            for n in train_names:  # random crops, ~4 per rally per epoch
                f, g = (feats[n] - mean) / std, targets[n]
                for _ in range(max(1, len(f) // L * 2)):
                    s = rng.integers(0, max(1, len(f) - L))
                    x, t = f[s:s + L], g[s:s + L]
                    if len(x) < L:
                        x = np.pad(x, ((0, L - len(x)), (0, 0)))
                        t = np.pad(t, (0, L - len(t)))
                    # augmentation: gain + noise (club microphones / crowds differ)
                    xs.append(x + rng.normal(0, 0.3) + rng.normal(0, 0.1, x.shape).astype(np.float32))
                    ys.append(t)
            idx = rng.permutation(len(xs))
            for i in range(0, len(idx), 32):
                b = idx[i:i + 32]
                x = torch.from_numpy(np.stack([xs[j] for j in b])).float().cuda()
                t = torch.from_numpy(np.stack([ys[j] for j in b])).float().cuda()
                loss = A.focal_loss(model(x), t)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model.eval(), mean, std

    def predict(model, mean, std, n):
        f = (feats[n] - mean) / std
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(f)[None].float().cuda())).cpu().numpy()[0]
        return A.frame_times(len(f)), p

    def best_thr(probs: dict, names_: list[str]) -> float:
        best = (0.0, 0.5)
        for thr in np.arange(0.2, 0.9, 0.05):
            tp = npred = ngt = 0
            for n in names_:
                t, p = probs[n]
                a, b, c = event_f1(list(A.peaks(t, p, thr)), [h[0] for h in hits[n]], 0.25)
                tp, npred, ngt = tp + a, npred + b, ngt + c
            best = max(best, (prf(tp, npred, ngt)["f1"], float(thr)))
        return best[1]

    oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fold_thr: dict[str, float] = {}
    res = {"folds": [sorted(f) for f in fold_sets], "per_fold": []}
    for k, test_t in enumerate(fold_sets):
        tr = [n for n in names if tournament(n) not in test_t]
        te = [n for n in names if tournament(n) in test_t]
        model, mean, std = train(tr)
        tr_probs = {n: predict(model, mean, std, n) for n in tr}
        thr = best_thr(tr_probs, tr)
        fr = {"test_tournaments": sorted(test_t), "threshold": thr}
        for collar in COLLARS:
            tp = npred = ngt = 0
            for n in te:
                t, p = predict(model, mean, std, n)
                oof[n] = (t, p)
                fold_thr[n] = thr
                a, b, c = event_f1(list(A.peaks(t, p, thr)), [h[0] for h in hits[n]], collar)
                tp, npred, ngt = tp + a, npred + b, ngt + c
            fr[f"collar_{int(collar * 1000)}ms"] = prf(tp, npred, ngt)
        res["per_fold"].append(fr)
        log("audio fold", k, json.dumps(fr)[:300])
    for collar in COLLARS:
        tp = npred = ngt = 0
        for n in names:
            t, p = oof[n]
            a, b, c = event_f1(list(A.peaks(t, p, fold_thr[n])), [h[0] for h in hits[n]], collar)
            tp, npred, ngt = tp + a, npred + b, ngt + c
        res[f"overall_collar_{int(collar * 1000)}ms"] = prf(tp, npred, ngt)
    # final model on every rally
    model, mean, std = train(names)
    all_probs = {n: predict(model, mean, std, n) for n in names}
    thr = best_thr(all_probs, names)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "mean": mean, "std": std, "threshold": thr,
                "sr": A.SR, "n_fft": A.N_FFT, "hop": A.HOP, "n_mels": A.N_MELS,
                "trained_on": "Decorte et al. CVPRW 2024 padel rallies (99 rallies, 11 tournaments)"},
               OUT / "models" / "audio_hit.pt")
    res["final_threshold"] = thr
    np.savez_compressed(OUT / "audio_oof.npz", **{f"t__{n}": oof[n][0] for n in names},
                        **{f"p__{n}": oof[n][1] for n in names}, **{f"thr__{n}": fold_thr[n] for n in names})
    return res, oof, fold_thr


# ---------------------------------------------------------------- B. video detections
def stage_detect(dec: Path, names: list[str]) -> dict:
    from padel_analyzer.engine.detect import Detector

    ball_w = str(first("/kaggle/input/**/models/ball_v2/best.pt") or first("/kaggle/input/**/models/ball_all/best.pt"))
    pose_w = str(first("/kaggle/input/**/models/pose_all/best.pt"))
    cache = OUT / "det"
    cache.mkdir(parents=True, exist_ok=True)
    errors = []

    def work(part, dev):
        try:
            det = Detector(ball_w, pose_w, device=f"cuda:{dev}", ball_imgsz=1536, pose_imgsz=1280, pose_stride=1)
            for n in part:
                p = cache / (n + ".npz")
                if not p.exists():
                    det.run(dec / "rallies" / n).save(p)
        except Exception:  # noqa: BLE001
            errors.append(traceback.format_exc()[-1500:])

    import torch

    g = max(1, torch.cuda.device_count())
    th = [threading.Thread(target=work, args=(names[i::g], i)) for i in range(g)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return {"ball": ball_w, "pose": pose_w, "errors": errors, "n": len(list(cache.glob("*.npz")))}


# ---------------------------------------------------------------- C/D. hit classifier
def stage_fusion(dec: Path, hits: dict, fold_sets, oof, fold_thr) -> dict:
    import lightgbm as lgb

    import train_events as T
    from padel_analyzer.engine import audio as A
    from padel_analyzer.engine import ball as B
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import events as E
    from padel_analyzer.engine import players as P
    from padel_analyzer.engine.detect import Detections
    from padel_analyzer.eval import pt100

    # every tournament uses the WPT main-camera framing; PT100's homography is the shared approximation
    calib = C.Calibration(pt100.load("F").H, (1920, 1080))
    names = sorted(hits)
    data = {}
    for n in names:
        det = Detections.load(OUT / "det" / (n + ".npz"))
        tracks = P.track(det, calib)
        bt = B.track_for_events(det.ball)
        t, p = oof[n]
        aud = A.video_features(t, p, det.n, det.fps, det.step, fold_thr[n])
        X, nearest = E.features(bt, tracks, aud)
        gt = [int(round(a * det.fps)) for a, _ in hits[n]]
        data[n] = {"X": X, "nearest": nearest, "y": E.labels(det.n, gt, radius=2), "gt": gt, "fps": det.fps,
                   "ball_obs": float(bt.observed.mean())}
    nv = len(E.VISUAL_FEATURES)
    variants = {"video": list(range(nv)), "audio": list(range(nv, len(E.FEATURES))),
                "fused": list(range(len(E.FEATURES)))}
    params = dict(T.HIT_PARAMS)

    def fit(train_names, cols):
        X = np.vstack([data[n]["X"][:, cols] for n in train_names])
        y = np.concatenate([data[n]["y"] for n in train_names])
        bst = lgb.train(params, lgb.Dataset(X, y), num_boost_round=400)
        best = (0.0, 0.5)
        for thr in np.arange(0.2, 0.9, 0.05):
            tp = npred = ngt = 0
            for n in train_names:
                hs = E.pick_peaks(bst.predict(data[n]["X"][:, cols]), data[n]["nearest"], thr, nms=8)
                a, b, c = event_f1([h.frame for h in hs], data[n]["gt"], 5)
                tp, npred, ngt = tp + a, npred + b, ngt + c
            best = max(best, (prf(tp, npred, ngt)["f1"], float(thr)))
        return bst, best[1]

    res = {"ball_observed_mean": round(float(np.mean([d["ball_obs"] for d in data.values()])), 4)}
    for v, cols in variants.items():
        tot = {k: [0, 0, 0] for k in TOL_FRAMES}
        per_t = {}
        for test_t in fold_sets:
            tr = [n for n in names if tournament(n) not in test_t]
            te = [n for n in names if tournament(n) in test_t]
            bst, thr = fit(tr, cols)
            for n in te:
                hs = E.pick_peaks(bst.predict(data[n]["X"][:, cols]), data[n]["nearest"], thr, nms=8)
                for k, tol in TOL_FRAMES.items():
                    a, b, c = event_f1([h.frame for h in hs], data[n]["gt"], tol)
                    tot[k][0] += a
                    tot[k][1] += b
                    tot[k][2] += c
                    if k == "200ms":
                        pt = per_t.setdefault(tournament(n), [0, 0, 0])
                        pt[0] += a
                        pt[1] += b
                        pt[2] += c
        res[v] = {k: prf(*x) for k, x in tot.items()}
        res[v]["per_tournament_200ms_f1"] = {t: prf(*x)["f1"] for t, x in sorted(per_t.items())}
        log("hits", v, json.dumps(res[v])[:300])

    # production fused model: all Decorte rallies + PT100 held-out rows (visual only, audio NaN)
    X = np.vstack([data[n]["X"] for n in names])
    y = np.concatenate([data[n]["y"] for n in names])
    for k in "FM":
        d = T.prepare(k, "heldout")
        X = np.vstack([X, d["X"]])
        y = np.concatenate([y, E.labels(d["n"], d["gt_hits"], radius=2)])
    bst = lgb.train(params, lgb.Dataset(X, y, feature_name=E.FEATURES), num_boost_round=400)
    _, thr = fit(names, variants["fused"])
    bst.save_model(str(OUT / "models" / "hit_lgbm.txt"))
    (OUT / "models" / "hit_lgbm.json").write_text(json.dumps({"threshold": thr, "source": "decorte+pt100 fused"}))
    res["production_threshold"] = thr
    return res


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "lightgbm"], check=False)
    OUT.mkdir(parents=True, exist_ok=True)
    _, dec = setup()
    hits = load_hits(dec)
    names = sorted(hits)
    fold_sets = folds(names)
    res: dict = {"rallies": len(names), "hits": sum(len(v) for v in hits.values()),
                 "tournaments": sorted({tournament(n) for n in names})}

    def save():
        (OUT / "audio.json").write_text(json.dumps(res, indent=1, default=str))

    t0 = time.time()
    try:
        res["audio"], oof, fold_thr = stage_audio(dec, hits, fold_sets)
    except Exception:  # noqa: BLE001
        res["audio"] = "FAILED " + traceback.format_exc()[-2000:]
        save()
        raise
    log("audio done", f"{(time.time() - t0) / 60:.1f} min")
    save()
    try:
        res["detect"] = stage_detect(dec, names)
    except Exception:  # noqa: BLE001
        res["detect"] = "FAILED " + traceback.format_exc()[-2000:]
    log("detect done", f"{(time.time() - t0) / 60:.1f} min")
    save()
    try:
        res["hits"] = stage_fusion(dec, hits, fold_sets, oof, fold_thr)
    except Exception:  # noqa: BLE001
        res["hits"] = "FAILED " + traceback.format_exc()[-2000:]
    save()
    log("final", json.dumps(res, default=str)[:3000])


if __name__ == "__main__":
    main()
