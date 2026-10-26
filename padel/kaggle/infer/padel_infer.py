"""Padel GPU inference kernel: analyse uploaded matches on Kaggle (the API's PADEL_BACKEND=kaggle).

Mounted:
  padel-jobs        job.json + videos (one dataset version per submission from the API worker)
  padel-code        padel_analyzer/
  padel-gpu-train   ball_all/best.pt          -> models/ball.pt
  padel-gpu-pose    pose_all/best.pt, *_lgbm  -> models/pose.pt, models/hit_lgbm.*, shot_lgbm.txt
  padel-stroke-rgb  stroke_rgb.pt             -> models/stroke_rgb.pt
Writes /kaggle/working/<match_id>/report.json (+ detections.npz) per job.
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


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def first(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return Path(hits[0]) if hits else None


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.189", "lightgbm"], check=False)
    code = first("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    repo = Path("/kaggle/working/padel")
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(code, repo)
    models = repo / "models"
    models.mkdir(exist_ok=True)
    wanted = {
        "ball.pt": ["/kaggle/input/**/models/ball_v2/best.pt",  # PT100 + Roboflow venues (padel-venues)
                    "/kaggle/input/**/models/ball_all/best.pt"],
        "pose.pt": "/kaggle/input/**/models/pose_all/best.pt",
        "hit_lgbm.txt": "/kaggle/input/**/out/models/hit_lgbm.txt",
        "hit_lgbm.json": "/kaggle/input/**/out/models/hit_lgbm.json",
        "shot_lgbm.txt": "/kaggle/input/**/out/models/shot_lgbm.txt",
        "stroke_rgb.pt": "/kaggle/input/**/stroke_rgb.pt",
    }
    for name, pat in wanted.items():
        src = next((p for p in map(first, pat if isinstance(pat, list) else [pat]) if p), None)
        if src:
            shutil.copy(src, models / name)
    log("models", sorted(p.name for p in models.iterdir()))
    sys.path.insert(0, str(repo))
    from padel_analyzer.engine import court as C
    from padel_analyzer.engine import pipeline
    from padel_analyzer.engine.detect import Detector

    jobs_file = first("/kaggle/input/**/padel-jobs/**/job.json") or first("/kaggle/input/**/job.json")
    jobs = json.loads(jobs_file.read_text())
    det = Detector(device="cuda", pose_stride=1)
    for job in jobs:
        out = Path("/kaggle/working") / job["match_id"]
        out.mkdir(parents=True, exist_ok=True)
        try:
            video = first(f"/kaggle/input/**/{job['video']}")
            log("job", job["match_id"], video)
            rep = pipeline.run(video, C.Calibration.from_dict(job["calibration"]), det,
                               max_frames=job.get("max_frames", 0), det_cache=out / "detections.npz",
                               progress=lambda s, p: None)
            rep["quality"]["models"] = sorted(p.name for p in models.iterdir())
            (out / "report.json").write_text(json.dumps(rep))
            log("done", job["match_id"], rep["match"]["rallies"], "rallies", rep["match"]["hits"], "hits")
        except Exception:  # noqa: BLE001
            (out / "error.txt").write_text(traceback.format_exc())
            log("FAILED", job["match_id"], traceback.format_exc()[-1500:])


if __name__ == "__main__":
    main()
