# CourtLens — facility-first futsal analytics

Cloud post-match analytics for futsal facilities. Docs: `PRD.md` (requirements), `ARCHITECTURE.md` (pipeline + stat definitions), `EVAL.md` (phase gates), `PILOT.md` (pilot plan).

## Phases in this repo

Phase numbering follows `PRD.md` §10; gates live in `EVAL.md`.

| Phase | What landed |
| --- | --- |
| **0 Eval + data** | `scripts/eval/*`, detection smoke, Roboflow download, corpus list; feasibility CLI `python -m app.phase0.benchmark` + UI `/phase0` |
| **1 Detect/track** | YOLO (`yolov5x_football.pt` default) + BoT-SORT (ByteTrack optional), OSNet ReID, ball gap interpolation, court format 5v5/6v6/7v7, thumbs |
| **2 Identity** | Torso Lab k-means kit split (Kit A/B, refs/outliers → Kit X), jersey OCR, per-player avatars, staff rename/reassign |
| **3 Possession** | Possession zones + gain/loss (`possession.py`) |
| **4 Events + stats** | Heuristic pass/shot/goal/assist/dribble, touches, possession %, per-player heatmaps |
| **5 Product + multi-cam** | Auth, facilities/roles, signed uploads (local or S3), jobs + cancel/retry, retention, review accept/reject/edit + audit, clips, publish + share tokens; cameras A–D with per-camera sync, cross-camera identity and fuse |

Event inference is **heuristic** (not a trained event model) so the review workflow is complete; lock quality after footage eval.

## Quick start

```bash
export DOCKER_HOST=unix:///var/run/docker.sock   # if needed
docker compose up --build
```

- App: http://localhost:5173  
- API: http://localhost:8000/docs  

Env knobs: `VISION_BACKEND=yolo|stub`, `YOLO_MODEL=yolov8m.pt` (default; finds 4–80× more futsal players than the football fine-tune — see `EVAL.md`), `VISION_DEVICE=cpu`, `VISION_FRAME_STRIDE=5`, `ALLOW_SINGLE_CAMERA=true`, `VISION_TARGET_FPS=10`, `VISION_TEAM_EMBED=1` (torso colour + SigLIP team split), `VISION_IMGSZ=0` (auto input size), `PIPELINE_PROXIES`, `PIPELINE_MAX_CLIPS`, `MIN_VIDEO_DURATION_S`. Court calibration (Setup → Court calibration) turns distance/speed into metres and km/h.

## GPU runs (Kaggle)

The Kaggle kernel runs the real backend pipeline (`app.scripts.run_offline`, SQLite + local storage) — no forked copy:

```bash
python scripts/kaggle/build_kaggle.py --push
kaggle kernels output muddasirr/courtlens-gpu-eval -p data/kaggle_out/gpu_eval
```

Details and gates: `EVAL.md`.

### Phase 0 CLI (no UI)

```bash
cd backend && source .venv/bin/activate
python -m app.phase0.benchmark /path/to/clip.mp4 --max-frames 200 --stride 5
```
