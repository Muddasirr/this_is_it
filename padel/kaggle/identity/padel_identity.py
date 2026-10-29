"""Kernel: pair identity across changes of ends — shirt histograms vs DINOv2 appearance embeddings.

Truth per segment (which scoreboard pair plays the far end) from the scoreboard serve dot + where the serve
was hit (labelled serve, else a detected serve), as in padel-points. For each feature type the production
linker (players._link_identity) is run and its far-pair identity per segment compared with the truth.
Saves features + truth (out/features_<match>.npz, out/truth_<match>.json) for CPU-only linker experiments.
"""
from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working/out")
REPO = Path("/kaggle/working/padel")


def first(p):
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
    for name, pat in {"hit_lgbm_video.txt": "/kaggle/input/**/padel-eval/out/models/hit_lgbm.txt",
                      "hit_lgbm_video.json": "/kaggle/input/**/padel-eval/out/models/hit_lgbm.json",
                      "hit_lgbm.txt": "/kaggle/input/**/padel-fusion/out/models/hit_lgbm.txt",
                      "hit_lgbm.json": "/kaggle/input/**/padel-fusion/out/models/hit_lgbm.json",
                      "shot_lgbm.txt": "/kaggle/input/**/out/models/shot_lgbm.txt",
                      "stroke_rgb.pt": "/kaggle/input/**/stroke_rgb.pt"}.items():
        src = first(pat)
        if src:
            shutil.copy(src, models / name)
    labels = first("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json").parent
    (REPO / "data" / "pt100").mkdir(parents=True, exist_ok=True)
    shutil.copytree(labels, REPO / "data" / "pt100" / "labels", dirs_exist_ok=True)
    sys.path.insert(0, str(REPO))


def dino_embeddings(video, det, every_s=0.5):
    """{(frame, det_index): L2-normalised DINOv2-S CLS embedding of the player crop}, same sampling as sample_hists."""
    import cv2
    import torch

    from padel_analyzer.engine import players as P

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    cap = cv2.VideoCapture(str(video))
    step = max(1, int(round(det.fps * every_s)))
    keys, crops, out = [], [], {}

    def flush():
        if not crops:
            return
        x = torch.from_numpy(np.stack(crops)).cuda().permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).half()
        with torch.no_grad():
            e = model(x).float()
        e = torch.nn.functional.normalize(e, dim=1).cpu().numpy()
        for k, v in zip(keys, e):
            out[k] = v
        keys.clear()
        crops.clear()

    for fv in range(det.n * det.step):
        if not cap.grab():
            break
        if fv % det.step:
            continue
        f = fv // det.step
        if f % step or not det.pose_done[f]:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for j in np.where(det.pose_box[f, :, 4] >= P.MIN_CONF)[0]:
            x1, y1, x2, y2 = det.pose_box[f, j, :4]
            w, h = x2 - x1, y2 - y1
            x1, x2 = int(max(0, x1 - 0.1 * w)), int(min(frame.shape[1], x2 + 0.1 * w))
            y1, y2 = int(max(0, y1 - 0.05 * h)), int(min(frame.shape[0], y2 + 0.05 * h))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (112, 224))[:, :, ::-1]
            keys.append((f, int(j)))
            crops.append(np.ascontiguousarray(crop))
            if len(crops) == 256:
                flush()
    flush()
    return out


def truth_segments(rep, segments, g, states, transitions):
    """segment index -> scoreboard row (0 top, 1 bottom) of the pair at the far end."""
    shots, rallies = rep["shots"], rep["rallies"]
    gt_serves = [s.frame for s in g.shots if s.kind == "serve"]
    seg_of = lambda f: next((i for i, (a, b) in enumerate(segments) if a <= f < b), len(segments) - 1)  # noqa: E731
    out = {}
    for t in transitions(states):
        if t["server_row"] is None:
            continue
        rs = [r for r in rallies if r["shots"] and t["from"] - 30 <= shots[r["shots"][-1]]["frame"] < t["to"]]
        if len(rs) != 1:
            continue
        first_f = shots[rs[0]["shots"][0]]["frame"]
        sf = next((f for f in gt_serves if first_f - 45 <= f <= first_f + 45), None)
        sp = None
        if sf is not None:
            pm = g.pos_m[sf]
            ok = ~np.isnan(pm[:, 0])
            if ok.any() and not np.isnan(g.ball[sf]).any():
                c = (g.pbox[sf][:, :2] + g.pbox[sf][:, 2:]) / 2
                d = np.where(ok, np.linalg.norm(c - g.ball[sf], axis=1), np.inf)
                sp = float(pm[int(np.argmin(d))][1])
        s0 = shots[rs[0]["shots"][0]]
        if sp is None and s0["type"] == "serve" and s0.get("pos_m"):
            sp = s0["pos_m"][1]
        if sp is None:
            continue
        out[seg_of(first_f)] = t["server_row"] if sp < 10 else 1 - t["server_row"]
    return out


def score(tracks, truth):
    o = {i: int(tracks.team[a, 0]) for i, (a, b) in enumerate(tracks.segments)}
    pairs = [(o[i], r) for i, r in truth.items() if i in o]
    agree = sum(a == b for a, b in pairs)
    return {"checked": len(pairs), "correct": max(agree, len(pairs) - agree)}


def main():
    setup()
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import pipeline
    from padel_analyzer.engine import players as P
    from padel_analyzer.engine.detect import Detections
    from padel_analyzer.eval import pt100
    from padel_analyzer.eval.scoreboard import transitions

    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for key in ("M", "F"):
        try:
            name = pt100.MATCHES[key]
            g = pt100.load(key)
            z = np.load(first(f"/kaggle/input/**/cache/{name}_heldout.npz"), allow_pickle=True)
            n = len(z["ball"])
            det = Detections(30.0, (1920, 1080), z["ball"].astype(np.float32), z["pose_box"].astype(np.float32),
                             z["pose_kpt"].astype(np.float32), np.ones(n, bool), np.zeros(n, bool))
            calib = C.Calibration(g.H, (1920, 1080))
            video = first(f"/kaggle/input/**/videos/{name}.mp4")
            hists = pipeline.sample_hists(video, det)
            log(key, "hists", len(hists))
            emb = dino_embeddings(video, det)
            log(key, "dino", len(emb))
            t_hist = P.track(det, calib, hists)
            t_dino = P.track(det, calib, emb)
            pipeline.sample_hists = lambda *a, **k: hists
            rep = pipeline.analyze(det, calib, video)
            states = json.loads(first(f"/kaggle/input/**/{name}/states.json").read_text())
            truth = truth_segments(rep, t_hist.segments, g, states, transitions)
            res[key] = {"segments": len(t_hist.segments), "shirt_hist": score(t_hist, truth), "dino": score(t_dino, truth)}
            log(key, json.dumps(res[key]))
            (OUT / f"truth_{key}.json").write_text(json.dumps({"segments": t_hist.segments, "far_row": truth}))
            ks = sorted(set(hists) & set(emb))
            np.savez_compressed(OUT / f"features_{key}.npz", keys=np.array(ks), hist=np.array([hists[k] for k in ks]),
                                dino=np.array([emb[k] for k in ks]).astype(np.float16))
        except Exception:  # noqa: BLE001
            log("FAILED", key, traceback.format_exc()[-2500:])
    (OUT / "identity.json").write_text(json.dumps(res, indent=1))
    log("final", json.dumps(res))


if __name__ == "__main__":
    main()
