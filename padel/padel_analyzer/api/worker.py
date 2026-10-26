"""In-process analysis worker: one job at a time (detection saturates the CPU/GPU)."""
from __future__ import annotations

import json
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..engine import court as C
from ..engine import pipeline
from ..engine.detect import Detector
from . import db
from .settings import BACKEND, DATA_DIR, DEVICE, MATCH_DIR, MAX_FRAMES, POSE_STRIDE

log = logging.getLogger("padel.worker")
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="padel-job")
_detector: Detector | None = None
_lock = threading.Lock()

# share of the progress bar per stage
STAGES = {"upload": (0.0, 0.15), "queued_gpu": (0.15, 0.2), "gpu": (0.2, 0.95), "download": (0.95, 1.0),
          "detect": (0.0, 0.8), "identity": (0.8, 0.92), "players": (0.92, 0.95), "ball": (0.95, 0.97),
          "hits": (0.97, 0.99), "rallies": (0.99, 1.0)}


def detector() -> Detector:
    global _detector
    with _lock:
        if _detector is None:
            _detector = Detector(device=DEVICE, pose_stride=POSE_STRIDE)
        return _detector


def _update(match_id: str, **fields) -> None:
    with db.Session() as s:
        m = s.get(db.Match, match_id)
        if m:
            for k, v in fields.items():
                setattr(m, k, v)
            s.commit()


def submit(match_id: str) -> None:
    _update(match_id, status="queued", progress=0.0, stage=None, error=None)
    _pool.submit(_run, match_id)


def _run(match_id: str) -> None:
    with db.Session() as s:
        m = s.get(db.Match, match_id)
        if not m or not m.calibration:
            return
        video = MATCH_DIR / m.id / m.filename
        calib = C.Calibration.from_dict(m.calibration)
    folder = MATCH_DIR / match_id
    last = {"p": -1.0}

    def progress(stage: str, p: float) -> None:
        a, b = STAGES.get(stage, (0, 1))
        val = a + (b - a) * max(0.0, min(1.0, p))
        if val - last["p"] >= 0.005 or p == 0:
            last["p"] = val
            _update(match_id, stage=stage, progress=val)

    try:
        _update(match_id, status="processing", stage="detect", progress=0.0)
        if BACKEND == "kaggle":
            report = _run_on_kaggle(match_id, video, calib, folder, progress)
        else:
            report = pipeline.run(video, calib, detector, max_frames=MAX_FRAMES, progress=progress,
                                  det_cache=folder / "detections.npz")
        (folder / "report.json").write_text(json.dumps(report))
        _update(match_id, status="done", stage=None, progress=1.0)
    except Exception as exc:  # noqa: BLE001
        log.exception("analysis failed")
        _update(match_id, status="failed", error=f"{exc}\n{traceback.format_exc()[-1500:]}")


def _run_on_kaggle(match_id: str, video: Path, calib: C.Calibration, folder: Path, progress) -> dict:
    """Upload video + calibration as a padel-jobs dataset version, run the padel-infer GPU kernel,
    download its report. The local machine only moves files."""
    import os
    import shutil
    import sys
    import time

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle"))
    import kaggle_cli as K

    job_dir = DATA_DIR / "kaggle_job"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    vname = f"{match_id}{video.suffix or '.mp4'}"
    try:
        os.link(video, job_dir / vname)  # no copy
    except OSError:
        shutil.copy(video, job_dir / vname)
    (job_dir / "job.json").write_text(json.dumps([{"match_id": match_id, "video": vname,
                                                   "calibration": calib.to_dict(), "max_frames": MAX_FRAMES}]))
    progress("upload", 0.0)
    if K.push_jobs(job_dir) != 0:
        raise RuntimeError("Kaggle dataset upload failed")
    for _ in range(240):  # dataset processing
        if K.dataset_ready(K.JOBS_DATASET):
            break
        time.sleep(15)
    progress("queued_gpu", 0.0)
    if K.kaggle("kernels", "push", "-p", str(K.write_metadata("infer"))) != 0:
        raise RuntimeError("Kaggle kernel push failed")
    t0 = time.time()
    time.sleep(30)
    while True:
        st = K.kernel_status("infer")
        if st in ("complete", "error", "cancel"):
            break
        # rough progress: a 30 fps match takes ~ duration x 1.5 on a T4
        progress("gpu", min(0.95, (time.time() - t0) / 3600))
        time.sleep(30)
    progress("download", 0.0)
    out = folder / "kaggle_out"
    K.download_output("infer", str(out), pattern=match_id)
    rep = next(out.rglob("report.json"), None)
    if rep is None:
        err = next(out.rglob("error.txt"), None)
        raise RuntimeError(f"Kaggle run {st}: " + (err.read_text()[-1500:] if err else "no report produced"))
    det = next(out.rglob("detections.npz"), None)
    if det:
        shutil.copy(det, folder / "detections.npz")
    return json.loads(rep.read_text())
