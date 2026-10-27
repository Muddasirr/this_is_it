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
from .settings import BACKEND, DATA_DIR, DEVICE, KAGGLE_JOB_TIMEOUT_S, MATCH_DIR, MAX_FRAMES, POSE_STRIDE

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


def recover_interrupted() -> None:
    """Jobs cannot survive a server restart (the worker is in-process): mark them failed, re-run from the UI."""
    with db.Session() as s:
        for m in s.query(db.Match).filter(db.Match.status.in_(("queued", "processing"))).all():
            m.status, m.stage = "failed", None
            m.error = "Interrupted by a server restart — press Re-run."
        s.commit()


def _retry(fn, what: str, tries: int = 4, wait_s: float = 20.0):
    import time

    for i in range(tries):
        try:
            rc = fn()
            if rc in (0, None, True):
                return rc
            err = f"exit code {rc}"
        except Exception as exc:  # noqa: BLE001 - network hiccups to Kaggle are common
            err = str(exc)
        log.warning("%s failed (%s), attempt %d/%d", what, err, i + 1, tries)
        time.sleep(wait_s * (i + 1))
    raise RuntimeError(f"{what} failed after {tries} attempts: {err}")


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

    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle"))
    import kaggle_cli as K

    K = importlib.reload(K)  # kernel sources / model locations change between jobs; never use a stale copy

    job_dir = DATA_DIR / "kaggle_job"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    # unique per run: the "is the new dataset version live" check below keys on this name
    vname = f"{match_id}_{int(time.time())}{video.suffix or '.mp4'}"
    try:
        os.link(video, job_dir / vname)  # no copy
    except OSError:
        shutil.copy(video, job_dir / vname)
    (job_dir / "job.json").write_text(json.dumps([{"match_id": match_id, "video": vname,
                                                   "calibration": calib.to_dict(), "max_frames": MAX_FRAMES}]))
    progress("upload", 0.0)
    _retry(lambda: K.push_jobs(job_dir), "Kaggle dataset upload")
    # wait until the new dataset version (with this match's video) is live: starting the GPU job earlier
    # mounts the previous version and silently analyses the previous match
    for _ in range(240):
        try:
            if K.dataset_ready(K.JOBS_DATASET) and K.dataset_has_file(K.JOBS_DATASET, vname):
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(15)
    else:
        raise RuntimeError("Kaggle never published the uploaded video (dataset version not ready after 60 min)")
    progress("queued_gpu", 0.0)
    _retry(lambda: K.kaggle("kernels", "push", "-p", str(K.write_metadata("infer"))), "Kaggle kernel push")
    t0 = time.time()
    time.sleep(30)
    import cv2

    cap = cv2.VideoCapture(str(video))
    expected_s = max(600.0, 1.6 * (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    cap.release()
    st = "unknown"
    started = False  # right after a push Kaggle may still report the *previous* run as complete
    while True:
        try:
            st = K.kernel_status("infer")
        except Exception:  # noqa: BLE001 - keep polling through network hiccups
            st = "unknown"
        if st in ("queued", "running"):
            started = True
        if st in ("complete", "error", "cancel") and (started or time.time() - t0 > 1800):
            break
        if time.time() - t0 > KAGGLE_JOB_TIMEOUT_S:
            raise RuntimeError(f"Kaggle GPU job exceeded {KAGGLE_JOB_TIMEOUT_S / 3600:.1f} h")
        # GPU time is ~1.5x the match duration on a T4 (detection + audio + stroke clips)
        progress("gpu", min(0.97, (time.time() - t0) / expected_s))
        time.sleep(30)
    progress("download", 0.0)
    out = folder / "kaggle_out"
    if out.exists():
        shutil.rmtree(out)  # a stale report from an earlier run must never be picked up
    _retry(lambda: K.download_output("infer", str(out), pattern=match_id), "Kaggle output download")
    rep = next(out.rglob("report.json"), None)
    if rep is None:
        err = next(out.rglob("error.txt"), None)
        raise RuntimeError(f"Kaggle run {st}: " + (err.read_text()[-1500:] if err else "no report produced"))
    det = next(out.rglob("detections.npz"), None)
    if det:
        shutil.copy(det, folder / "detections.npz")
    return json.loads(rep.read_text())
