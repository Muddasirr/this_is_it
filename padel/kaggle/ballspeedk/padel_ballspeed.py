"""CPU kernel: ball speed (engine/ballspeed.py) on PadelTracker100 + Kuwait 2025.

  labelled  PT100 hand-labelled ball positions at the labelled hits (hitter = labelled player nearest the ball)
  detected  our held-out ball track at the same hits (same hitter feet: isolates the ball detection)
  kuwait    our report's hits + ball track (unseen tournament)
No radar truth exists for padel video, so checks are physical: per stroke type, coverage and speed
distribution, and labelled-vs-detected agreement on the same shots.
"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

code = Path(sorted(glob.glob("/kaggle/input/**/padel_analyzer/__init__.py", recursive=True))[0]).parents[1]
repo = Path("/tmp/padel")
shutil.copytree(code, repo)
labels = Path(sorted(glob.glob("/kaggle/input/**/pt100/labels/2022_BCN_FinalF_1_pose.json", recursive=True))[0]).parent
(repo / "data" / "pt100").mkdir(parents=True, exist_ok=True)
shutil.copytree(labels, repo / "data" / "pt100" / "labels")
sys.path.insert(0, str(repo))

from padel_analyzer.engine import ball as B  # noqa: E402
from padel_analyzer.engine import ballspeed as BS  # noqa: E402
from padel_analyzer.engine import court as C  # noqa: E402
from padel_analyzer.engine.detect import Detections  # noqa: E402
from padel_analyzer.eval import pt100  # noqa: E402

OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)


def serve_box(rows):
    sv = [r for r in rows if r["type"] == "serve" and r["fit"]]
    ok = sum(1 for r in sv if ((10 < r["fit"]["bounce_m"][1] <= 17.25) if r["far"] else (2.75 <= r["fit"]["bounce_m"][1] < 10)))
    return [ok, len(sv)]


def summary(rows):
    out = {}
    for k in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == k]
        v = np.array([r["speed_kmh"] for r in rs if r["speed_kmh"] is not None])
        out[k] = {"shots": len(rs), "with_speed": int(len(v)),
                  "p10_p50_p90": np.percentile(v, [10, 50, 90]).round(0).tolist() if len(v) else None}
    v = np.array([r["speed_kmh"] for r in rows if r["speed_kmh"] is not None])
    out["all"] = {"shots": len(rows), "with_speed": int(len(v)),
                  "p10_p50_p90": np.percentile(v, [10, 50, 90]).round(0).tolist() if len(v) else None}
    return out


res = {}
for key in ("F", "M"):
    g = pt100.load(key)
    calib = C.Calibration(g.H, (1920, 1080))
    cam = BS.camera_from_calibration(calib)
    res[f"camera_{key}"] = {"centre_m": cam.centre.round(2).tolist(), "focal_px": round(float(cam.K[0, 0])),
                            "floor_err_px": round(cam.floor_err_px, 2), "mirror": cam.mirror}
    n = g.shot_range[1]
    shots = [s for s in g.shots if s.frame < n]
    gt_obs = ~np.isnan(g.ball[:, 0])
    z = np.load(sorted(glob.glob(f"/kaggle/input/**/cache/{pt100.MATCHES[key]}_heldout.npz", recursive=True))[0])
    nn = len(z["ball"])
    det = Detections(30.0, (1920, 1080), z["ball"].astype(np.float32), z["pose_box"].astype(np.float32),
                     z["pose_kpt"].astype(np.float32), np.ones(nn, bool), np.zeros(nn, bool))
    bt = B.track(det.ball)
    lab, dtc = [], []
    for i, s in enumerate(shots):
        end = shots[i + 1].frame - 2 if i + 1 < len(shots) else s.frame + 75
        end = min(end, s.frame + 75)
        bx = g.ball[s.frame] if gt_obs[s.frame] else next((g.ball[k] for k in (s.frame - 1, s.frame + 1) if gt_obs[k]), None)
        pm = g.pos_m[s.frame]
        ok = ~np.isnan(pm[:, 0])
        feet = None
        if bx is not None and ok.any():
            c = (g.pbox[s.frame][:, :2] + g.pbox[s.frame][:, 2:]) / 2
            d = np.where(ok, np.linalg.norm(c - bx, axis=1), np.inf)
            feet = pm[int(np.argmin(d))]
        r = BS.shot_speed(cam, calib, g.ball, gt_obs, s.frame, end, feet, 30.0) if feet is not None else None
        lab.append({"frame": s.frame, "type": s.kind, "speed_kmh": r["speed_kmh"] if r else None, "fit": r,
                    "far": bool(feet is not None and feet[1] < C.NET_Y)})
        # detected: our held-out ball track, same hitter position
        rd = BS.shot_speed(cam, calib, bt.xy, bt.observed, s.frame, end, feet, 30.0) if feet is not None else None
        dtc.append({"frame": s.frame, "type": s.kind, "speed_kmh": rd["speed_kmh"] if rd else None, "fit": rd,
                    "far": bool(feet is not None and feet[1] < C.NET_Y)})
    both = [(a["speed_kmh"], b["speed_kmh"]) for a, b in zip(lab, dtc) if a["speed_kmh"] and b["speed_kmh"]]
    diff = np.array([abs(a - b) for a, b in both])
    res[key] = {"labelled_ball": summary(lab), "detected_ball": summary(dtc),
                "same_shot_abs_diff_kmh": {"n": len(both), "median": round(float(np.median(diff)), 1) if len(diff) else None,
                                           "p90": round(float(np.percentile(diff, 90)), 1) if len(diff) else None},
                "serves_in_box": {nm: serve_box(rows) for nm, rows in (("labelled", lab), ("detected", dtc))},
                "contact_h_by_type": {k: round(float(np.median([r["fit"]["contact_h_m"] for r in lab if r["fit"] and r["type"] == k])), 2)
                                      for k in {r["type"] for r in lab if r["fit"]}}}
    print(key, json.dumps(res[key]), flush=True)

# Kuwait: our hits + our ball track
kd = Path(sorted(glob.glob("/kaggle/input/**/kuwait_points.json", recursive=True))[0]).parent
rep = json.loads((kd / "report.json").read_text())
det = Detections.load(kd / "detections.npz")
calib = C.Calibration.from_dict(rep["calibration"])
cam = BS.camera_from_calibration(calib)
res["camera_K"] = {"centre_m": cam.centre.round(2).tolist(), "focal_px": round(float(cam.K[0, 0])),
                   "floor_err_px": round(cam.floor_err_px, 2)}
bt = B.track(det.ball)
sh = rep["shots"]
rows = []
for i, s in enumerate(sh):
    end = min(sh[i + 1]["frame"] - 2 if i + 1 < len(sh) else s["frame"] + 75, s["frame"] + 75)
    r = BS.shot_speed(cam, calib, bt.xy, bt.observed, s["frame"], end, np.array(s["pos_m"]), det.fps) if s.get("pos_m") else None
    rows.append({"type": s["type"], "speed_kmh": r["speed_kmh"] if r else None})
res["K"] = summary(rows)
print("K", json.dumps(res["K"]), flush=True)
(OUT / "ballspeed.json").write_text(json.dumps(res, indent=1))
print("final", json.dumps({k: v for k, v in res.items() if k.startswith("camera")}))
