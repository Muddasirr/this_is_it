"""CourtLens GPU jobs on Modal (second GPU source next to Kaggle).

Data and results live in the Modal volume "courtlens" (/data): stage inputs with
data/modal_stage/*.sh (downloads happen on this machine — no Kaggle key on Modal).
Training checkpoints are written to the volume every epoch, so a stopped run
resumes where it left off; `modal volume get courtlens <path>` brings weights home.

  modal run scripts/modal/courtlens_gpu.py --task ball          # train ball detector + eval
  modal run scripts/modal/courtlens_gpu.py --task ball-bench    # DFL pass benchmark with it
"""

from __future__ import annotations

from pathlib import Path

import modal

# Inside the Modal container this file sits at /root/<name>.py: the repo root only matters locally (image build)
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")
app = modal.App("courtlens")
vol = modal.Volume.from_name("courtlens", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    # transformers ≥ 5 (SigLIP team split / role filter) needs torch ≥ 2.6 — with 2.5 it silently
    # disables PyTorch and the GPU run loses kits + bystander filtering
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "ultralytics==8.3.40")
    .pip_install_from_requirements(str(ROOT / "backend" / "requirements.txt"))
    .pip_install("motmetrics==1.4.0", "numpy<2")  # ultralytics 8.3.40 still calls np.trapz (gone in NumPy 2)
    # Learned event spotter (app/services/action_spotting.py): T-DEED, GPL-3.0, server-side only, pinned
    .pip_install("SoccerNet==0.1.62", "huggingface_hub==0.26.2", "tabulate")
    .run_commands("git clone -q https://github.com/arturxe2/T-DEED.git /root/tdeed && "
                  "git -C /root/tdeed checkout -q d953b4de8a2ad941c565e1683c5b07255cbe644e")
    .env({"TDEED_REPO": "/root/tdeed", "TDEED_WEIGHTS": "/mnt/data/models/tdeed_soccertrack.pt"})
    .add_local_dir(str(ROOT / "backend" / "app"), "/root/backend/app")
    .add_local_dir(str(ROOT / "backend" / "cfg"), "/root/backend/cfg")
)
BACKEND = "/root/backend"


def _env(**extra) -> dict:
    import os

    return dict(os.environ, PYTHONPATH=BACKEND, VISION_DEVICE="0", VISION_TARGET_FPS="10", VISION_IMGSZ="0",
                YOLO_MODEL="yolov8m.pt", **extra)


@app.function(image=image, gpu="L4", volumes={"/mnt": vol}, timeout=8 * 3600)
def train_ball(epochs: int = 40) -> dict:
    import sys

    sys.path.insert(0, BACKEND)
    from app.scripts.train_ball import main

    return main(["--data", "/mnt/data/ball", "--out", "/mnt/data/ball_run", "--epochs", str(epochs)], on_epoch=vol.commit)


@app.function(image=image, gpu="L4", volumes={"/mnt": vol}, timeout=4 * 3600)
def ball_bench() -> dict:
    import json
    import subprocess
    import sys

    weights = "/mnt/data/ball_run/runs/ball/weights/best.pt"
    out = Path("/mnt/data/results/bench_events_ball.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "app.scripts.bench_events", "--clips", "/mnt/data/dfl_clips", "--out", str(out),
                        "--per-class", "150", "--save-dets", "/mnt/data/results/bench_dets_ball"], cwd=BACKEND,
                       env=_env(VISION_BALL_MODEL=weights, VISION_JERSEY_OCR="0", VISION_TEAM_EMBED="0"),
                       capture_output=True, text=True)
    Path("/mnt/data/results/bench_events_ball.log").write_text(r.stdout + "\n" + r.stderr)
    vol.commit()
    return json.loads(out.read_text())["pass_vs_not"] if out.exists() else {"error": r.stderr[-2000:]}


@app.function(image=image, gpu="L4", volumes={"/mnt": vol}, timeout=3 * 3600)
def detect_video(video: str, camera_label: str, env: dict) -> dict:
    """Product detect + track on GPU for the worker's ModalTrackBackend (VISION_BACKEND=modal)."""
    import dataclasses
    import io
    import os
    import sys
    import zipfile

    os.environ.update({**_env(VISION_BACKEND="yolo"), **env, "VISION_DEVICE": "0",
                       "VISION_THUMB_DIR": f"/tmp/thumbs_{Path(video).parent.name}"})
    sys.path.insert(0, BACKEND)
    from app.core.config import get_settings

    get_settings.cache_clear()
    from app.services.yolo_backend import YoloTrackBackend

    backend = YoloTrackBackend()
    dets = backend.detect_and_track(video, camera_label)
    buf = io.BytesIO()
    thumbs = Path(os.environ["VISION_THUMB_DIR"]) / camera_label
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in thumbs.glob("*.jpg") if thumbs.is_dir() else []:
            zf.write(p, p.name)
    Path(video).unlink(missing_ok=True)  # the uploaded copy is no longer needed
    vol.commit()
    return {"detections": [dataclasses.asdict(d) for d in dets], "stats": backend.last_run_stats,
            "thumbs_zip": buf.getvalue(), "spotted": backend.spotted.get(camera_label, [])}


@app.function(image=image, gpu="L4", volumes={"/mnt": vol}, timeout=4 * 3600)
def mot_bench(max_frames: int = 600) -> dict:
    """SoccerTrack MOT clips with OSNet ReID on: raw tracker vs offline stitching (roadmap W3)."""
    import json
    import subprocess
    import sys

    out_dir = Path("/mnt/data/results/mot")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for gt in sorted(Path("/mnt/data/mot").glob("*.txt")):
        clip = gt.parent / "clips" / f"{gt.stem}.mp4"
        teams = Path(BACKEND) / "app/scripts/bench_labels" / f"{gt.stem}_teams.json"
        out = out_dir / f"bench_mot_{gt.stem}.json"
        cmd = [sys.executable, "-m", "app.scripts.bench_mot", "--clip", str(clip), "--gt", str(gt), "--imgsz", "1920",
               "--max-frames", str(max_frames), "--reid", "--save-log", str(out_dir / "logs"), "--out", str(out)]
        if teams.exists():
            cmd += ["--teams", str(teams)]
        r = subprocess.run(cmd, cwd=BACKEND, env=_env(), capture_output=True, text=True)
        (out_dir / f"bench_mot_{gt.stem}.log").write_text(r.stdout + "\n" + r.stderr)
        results[gt.stem] = json.loads(out.read_text())["runs"] if out.exists() else {"error": r.stderr[-1500:]}
    vol.commit()
    return results


@app.function(image=image, gpu="L4", volumes={"/mnt": vol}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("courtlens-hf")])
def soccertrack_product_eval(match: str = "117093", minutes: float = 12.0) -> dict:
    """Whole product pipeline (run_offline: detect → spot → identity → events) on a SoccerTrack half,
    scored with bench_soccertrack — the honest number for the learned spotter (crop follows our own
    detections, never the annotated ball)."""
    import json
    import os
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download

    tok = os.environ["HF_TOKEN"]
    video = hf_hub_download("atomscott/soccertrack-v2", f"videos/{match}/{match}_panorama_1st_half.mp4",
                            repo_type="dataset", token=tok, local_dir="/tmp/hf")
    events = hf_hub_download("atomscott/soccertrack-v2", f"bas/{match}/{match}_12_class_events.json",
                             repo_type="dataset", token=tok, local_dir="/tmp/hf")
    out = Path(f"/mnt/data/results/product_{match}_{int(minutes)}min")
    frames = str(int(minutes * 60 * 10))  # VISION_TARGET_FPS=10
    r = subprocess.run([sys.executable, "-m", "app.scripts.run_offline", "--out", str(out), "--title", f"st_{match}",
                        "--camera", f"A={video}"], cwd=BACKEND,
                       env=_env(VISION_MAX_FRAMES=frames, MIN_VIDEO_DURATION_S="20"), capture_output=True, text=True)
    (out.parent / f"product_{match}.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    if r.returncode != 0:
        vol.commit()
        return {"error": r.stderr[-3000:]}
    s = subprocess.run([sys.executable, "-m", "app.scripts.bench_soccertrack", "--run", str(out), "--events", events,
                        "--half", "1", "--out", str(out / "bench_soccertrack.json")], cwd=BACKEND, env=_env(),
                       capture_output=True, text=True)
    vol.commit()
    summary = json.loads((out / "summary.json").read_text())
    stats = (summary.get("stages", {}).get("detect_track", {}).get("vision_stats") or [{}])[0]
    return {"bench": json.loads((out / "bench_soccertrack.json").read_text()) if (out / "bench_soccertrack.json").exists()
            else s.stderr[-2000:], "spotted_events": stats.get("spotted_events"), "stage_seconds": stats.get("stage_seconds")}


@app.local_entrypoint()
def main(task: str = "ball", epochs: int = 40):
    import json

    if task == "ball":
        print(json.dumps(train_ball.remote(epochs), indent=1)[:4000])
    elif task == "ball-bench":
        print(json.dumps(ball_bench.remote(), indent=1))
    elif task == "mot":
        print(json.dumps(mot_bench.remote(), indent=1)[:6000])
    else:
        raise SystemExit(f"unknown task {task}")
