"""Vision on a free Kaggle GPU (roadmap W1): VISION_BACKEND=kaggle.

Per camera: the video plus this worker's own backend code (so the GPU runs exactly
the code that will post-process its output) go up as a private, temporary Kaggle
dataset; a reusable runner kernel (KAGGLE_RUNNER) does detect + track + learned
event spotting on the T4 and writes detections, raw spotter output, run stats and the
player crops to one result zip; the worker downloads it, deletes the job dataset, and
continues the pipeline (identity, events, stats, review) as usual. Any failure falls
back to local YOLO — a job never fails because of Kaggle.

Latency is minutes of upload + queue + GPU time: fine for post-match processing, one
job at a time (one runner kernel). Quota: 30 GPU-hours / week per Kaggle account.
Credentials: ~/.kaggle/kaggle.json (mounted read-only into the worker).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.vision import Detection, EventCandidate, VisionBackend

logger = logging.getLogger(__name__)

RUNNER = os.environ.get("KAGGLE_RUNNER", "courtlens-job-runner")
MODELS_DATASET = os.environ.get("KAGGLE_MODELS_DATASET", "courtlens-futsal-models")
POLL_S = 30
MAX_WAIT_S = 6 * 3600
# In-flight job records (shared volume), so a restarted worker re-attaches to its GPU run
STATE_DIR = Path(os.environ.get("KAGGLE_STATE_DIR", "/data/videos/.kaggle_jobs"))
# Settings forwarded to the GPU run so it behaves like a local run (plus the spotter config)
FORWARD_ENV = ("YOLO_MODEL", "VISION_TARGET_FPS", "VISION_FRAME_STRIDE", "VISION_CONF", "VISION_IMGSZ",
               "VISION_IMGSZ_MAX", "VISION_MAX_FRAMES", "VISION_IMGSZ_FOOTBALL", "VISION_FOOTBALL_MAX_W", "VISION_TRACKER", "VISION_GMC", "VISION_REID",
               "VISION_TEAM_EMBED", "VISION_ROLE_FILTER", "VISION_JERSEY_OCR", "COURT_FORMAT", "TDEED_THRESHOLDS")
TDEED_COMMIT = "d953b4de8a2ad941c565e1683c5b07255cbe644e"

RUNNER_SCRIPT = r'''
import json, os, subprocess, sys, zipfile, dataclasses
from pathlib import Path
job = json.loads(next(Path("/kaggle/input").rglob("job.json")).read_text())
work = Path("/kaggle/working"); out = work / "out"; out.mkdir(exist_ok=True)
# Kaggle unpacks uploaded zips: backend.zip arrives as a folder holding app/ + cfg/ (or, rarely, as the zip)
bundle = next(iter(Path("/kaggle/input").rglob("backend.zip")), None)
if bundle is not None and bundle.is_file():
    zipfile.ZipFile(bundle).extractall(work / "backend")
else:
    code = next(Path("/kaggle/input").rglob("app/services/yolo_backend.py")).parents[2]
    subprocess.check_call(["cp", "-r", str(code), str(work / "backend")])
# Same set as the proven eval kernel (scripts/kaggle/build_kaggle.py PIP) + the spotter's deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.183", "lap>=0.5.12",
                       "pydantic-settings==2.6.1", "pytesseract>=0.3.10", "torchreid==0.2.5", "gdown>=5.1",
                       "umap-learn>=0.5.6", "SoccerNet==0.1.62", "tabulate"])
if not (work / "tdeed").exists():
    subprocess.call(f"git clone -q https://github.com/arturxe2/T-DEED.git {work}/tdeed && "
                    f"git -C {work}/tdeed checkout -q {job['tdeed_commit']}", shell=True)
weights = next(iter(Path("/kaggle/input").rglob("tdeed_futsal.pt")), None)
football = next(iter(Path("/kaggle/input").rglob("tdeed_soccertrack.pt")), None)  # used when COURT_FORMAT=11v11
detector = next(iter(Path("/kaggle/input").rglob("yolov5x_football.pt")), None)  # 11v11 people + referees
pano_ball = next(iter(Path("/kaggle/input").rglob("ball_st2.pt")), None)  # 11v11 panorama ball
os.environ.update(job["env"])
if detector is not None:
    os.environ.update(YOLO_MODEL_FOOTBALL=str(detector))
if pano_ball is not None:
    os.environ.update(VISION_BALL_MODEL_PANORAMA=str(pano_ball))
os.environ.update(VISION_DEVICE="0", VISION_BACKEND="yolo", VISION_THUMB_DIR=str(work / "thumbs"), FFMPEG="ffmpeg")
if weights is not None and (work / "tdeed").exists():
    os.environ.update(TDEED_REPO=str(work / "tdeed"), TDEED_WEIGHTS=str(weights))
if football is not None and (work / "tdeed").exists():
    os.environ.update(TDEED_REPO=str(work / "tdeed"), TDEED_WEIGHTS_FOOTBALL=str(football))
sys.path.insert(0, str(work / "backend"))
from app.services.yolo_backend import YoloTrackBackend
video = str(next(Path("/kaggle/input").rglob(job["video"])))
backend = YoloTrackBackend()
dets = backend.detect_and_track(video, job["camera"])
res = {"detections": [dataclasses.asdict(d) for d in dets], "stats": backend.last_run_stats,
       "spotted": backend.spotted.get(job["camera"], []), "job": job["id"]}
with zipfile.ZipFile(work / "result.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("result.json", json.dumps(res))
    thumbs = work / "thumbs" / job["camera"]
    for p in thumbs.glob("*.jpg") if thumbs.is_dir() else []:
        zf.write(p, f"thumbs/{p.name}")
print("done", len(dets), "detections,", len(res["spotted"]), "spotted")
'''


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _code_zip(dest: Path) -> None:
    """The worker's own app/ + cfg/ — the GPU run uses exactly this code."""
    root = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("app", "cfg"):
            for p in sorted((root / sub).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    zf.write(p, p.relative_to(root))


class KaggleTrackBackend(VisionBackend):
    name = "kaggle-yolo-botsort"
    version = "0.1.0"

    def __init__(self) -> None:
        self.last_run_stats: dict[str, Any] = {}
        self._phases: dict[str, float] = {}
        self.spotted: dict[str, list[dict[str, Any]]] = {}

    def detect_and_track(self, video_path: str, camera_label: str) -> list[Detection]:
        try:
            return self._remote(video_path, camera_label)
        except Exception as exc:  # noqa: BLE001 — quota / outage / auth: never fail the job for it
            logger.warning("Kaggle GPU run failed (%s) — falling back to local YOLO", exc)
            from app.services.yolo_backend import YoloTrackBackend

            local = YoloTrackBackend()
            dets = local.detect_and_track(video_path, camera_label)
            self.spotted[camera_label] = local.spotted.get(camera_label, [])
            self.last_run_stats = {**local.last_run_stats, "backend": "yolo-local-fallback", "kaggle_error": str(exc)[:300]}
            return dets

    def _remote(self, video_path: str, camera_label: str) -> list[Detection]:
        api = _api()
        owner = api.get_config_value("username")
        job_id = uuid.uuid4().hex[:10]
        ds_slug = f"courtlens-job-{job_id}"
        t0 = time.time()
        phases: dict[str, float] = {}  # wall seconds per phase → last_run_stats["kaggle_phases_s"]
        last = [t0]

        def mark(name: str) -> None:
            now = time.time()
            phases[name] = round(now - last[0], 1)
            last[0] = now

        kernel = f"{owner}/{RUNNER}"
        state = _state_file(video_path, camera_label)
        prior = _resumable(api, kernel, state)
        with tempfile.TemporaryDirectory() as tmp:
            if prior:  # a worker restart mid-job: re-attach to the GPU run instead of paying for it twice
                job_id, ds_slug = prior["job_id"], prior["dataset"]
                logger.info("Kaggle job %s already pushed for this video — re-attaching", job_id)
            else:
                job_id, ds_slug = self._push(api, owner, kernel, video_path, camera_label, job_id, ds_slug, Path(tmp), mark)
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps({"job_id": job_id, "dataset": ds_slug, "pushed_at": time.time()}))
            self._wait(lambda: _status(api, kernel) in ("complete", "error", "cancelacknowledged", "cancelrequested"),
                       "kernel", poll_s=15)
            mark("run")
            status = _status(api, kernel)
            out_dir = Path(tmp) / "out"
            _retry(api.kernels_output, kernel, path=str(out_dir), file_pattern=r"result\.zip$", page_size=200)
            result_zip = out_dir / "result.zip"
            mark("download")
            state.unlink(missing_ok=True)
            logger.info("Kaggle job %s phases (s): %s", job_id, phases)
            self._phases = phases
            if status != "complete" or not result_zip.exists():
                raise RuntimeError(f"runner {status}; result.zip present: {result_zip.exists()}")
            try:
                api.dataset_delete(owner, ds_slug, no_confirm=True)
            except Exception as exc:  # noqa: BLE001 — cleanup only
                logger.info("could not delete %s (%s)", ds_slug, exc)
            return self._parse(result_zip.read_bytes(), camera_label, job_id, time.time() - t0)

    def _push(self, api, owner: str, kernel: str, video_path: str, camera_label: str, job_id: str, ds_slug: str,
              tmp: Path, mark) -> tuple[str, str]:
        """Upload video + code as the job dataset and start the runner on it."""
        ds = Path(tmp) / "ds"
        ds.mkdir()
        video_name = f"{camera_label}{Path(video_path).suffix or '.mp4'}"
        shutil.copyfile(video_path, ds / video_name)
        _code_zip(ds / "backend.zip")
        job = {"id": job_id, "camera": camera_label, "video": video_name, "tdeed_commit": TDEED_COMMIT,
               "env": {k: os.environ[k] for k in FORWARD_ENV if os.environ.get(k)}}
        (ds / "job.json").write_text(json.dumps(job))
        (ds / "dataset-metadata.json").write_text(json.dumps(
            {"title": ds_slug, "id": f"{owner}/{ds_slug}", "licenses": [{"name": "other"}]}))
        _retry(api.dataset_create_new, str(ds), public=False, quiet=True, dir_mode="skip")
        mark("upload")
        self._wait(lambda: "ready" in str(api.dataset_status(f"{owner}/{ds_slug}")).lower(), "dataset", poll_s=10)
        mark("dataset_ready")

        kdir = Path(tmp) / "kernel"
        kdir.mkdir()
        (kdir / "runner.py").write_text(RUNNER_SCRIPT)
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{owner}/{RUNNER}", "title": RUNNER, "code_file": "runner.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{owner}/{ds_slug}", f"{owner}/{MODELS_DATASET}"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}))
        _retry(api.kernels_push, str(kdir))
        mark("push")
        # The previous session can still read "complete" right after the push: wait for this one to start
        started = time.time()
        while _status(api, kernel) in ("complete", "error", "cancelacknowledged") and time.time() - started < 600:
            time.sleep(10)
        mark("queue")
        return job_id, ds_slug

    def _parse(self, blob: bytes, camera_label: str, job_id: str, seconds: float) -> list[Detection]:
        thumb_root = Path(get_settings().vision_thumb_dir or Path(tempfile.gettempdir()) / "courtlens_thumbs")
        thumb_dir = thumb_root / camera_label
        thumb_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            res = json.loads(zf.read("result.json"))
            if res.get("job") and res["job"] != job_id:
                raise RuntimeError(f"runner output is job {res['job']}, expected {job_id}")
            for name in zf.namelist():
                if name.startswith("thumbs/") and name.endswith(".jpg"):
                    (thumb_dir / Path(name).name).write_bytes(zf.read(name))
        dets = []
        for d in res["detections"]:
            feats = d.get("features") or {}
            if feats.get("thumb_path"):
                feats["thumb_path"] = str(thumb_dir / Path(feats["thumb_path"]).name)
            dets.append(Detection(**{**d, "features": feats}))
        self.spotted[camera_label] = res.get("spotted") or []
        self.last_run_stats = {**(res.get("stats") or {}), "backend": "kaggle", "kaggle_job": job_id,
                               "kaggle_round_trip_s": round(seconds), "kaggle_phases_s": self._phases}
        return dets

    @staticmethod
    def _wait(done, what: str, poll_s: float = POLL_S) -> None:
        t0 = time.time()
        while True:
            try:
                if done():
                    return
            except Exception as exc:  # noqa: BLE001 — a fresh dataset answers 403 for its first seconds
                logger.info("Kaggle %s not visible yet (%s)", what, exc)
            if time.time() - t0 > MAX_WAIT_S:
                raise TimeoutError(f"Kaggle {what} not ready after {MAX_WAIT_S} s")
            time.sleep(poll_s)

    def infer_events(self, detections: list[Detection], camera_label: str) -> list[EventCandidate]:
        from app.services.events import events_from_spotting, infer_event_candidates

        if self.spotted.get(camera_label):
            return events_from_spotting(self.spotted[camera_label], detections, camera_label)
        return infer_event_candidates(detections, camera_label)


def _transient(exc: Exception) -> bool:
    """Network blips (DNS, resets, timeouts) and Kaggle's 429 / 5xx — worth retrying, unlike 4xx."""
    import requests

    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return code == 429 or (code is not None and code >= 500)


def _retry(fn, *args, attempts: int = 8, **kwargs):
    """fn(*args, **kwargs), riding out transient failures for ~10 min (15 s backoff doubling, ≤ 120 s)."""
    delay = 15
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1 or not _transient(exc):
                raise
            logger.info("Kaggle call %s failed (%s) — retry in %d s", getattr(fn, "__name__", fn), exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def _state_file(video_path: str, camera_label: str) -> Path:
    """Where the in-flight Kaggle job for this video is recorded (survives worker restarts)."""
    h = hashlib.sha1(f"{camera_label}:{os.path.getsize(video_path)}".encode())
    with open(video_path, "rb") as f:
        h.update(f.read(4 << 20))
    return STATE_DIR / f"{h.hexdigest()[:16]}.json"


def _resumable(api, kernel: str, state: Path) -> dict | None:
    """The recorded job for this video if the runner is still on it (or just finished it)."""
    try:
        prior = json.loads(state.read_text())
    except (OSError, ValueError):
        return None
    if time.time() - prior.get("pushed_at", 0) > MAX_WAIT_S:
        return None
    try:
        return prior if _status(api, kernel) in ("queued", "running", "complete") else None
    except Exception:  # noqa: BLE001 — can't tell ⇒ start fresh
        return None


def _status(api, kernel: str) -> str:
    """'running' / 'queued' / 'complete' / 'error' … from KernelWorkerStatus.X"""
    r = _retry(api.kernels_status, kernel)
    return str(getattr(r, "status", r)).rsplit(".", 1)[-1].strip().lower()
