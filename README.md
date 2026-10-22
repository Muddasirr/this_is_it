# CourtLens — facility-first futsal analytics

Cloud post-match analytics for futsal facilities (PRD Phases 0–3 implemented in code).

## Phases in this repo

| Phase | What landed |
| --- | --- |
| **0 Feasibility** | YOLO detect/track (`ultralytics`), CLI `python -m app.phase0.benchmark`, UI `/phase0`, throughput + visibility report |
| **1 Facility foundation** | Auth, facilities/roles, teams, players, roster, signed uploads (local or S3), sync offset, jobs + cancel/retry, retention policy + enforce |
| **2 Vision + identity** | Per-camera tracks, model identity proposals, staff correction UI/API |
| **3 Events + review** | Heuristic goal/shot/pass candidates, fuse, review accept/reject/edit + audit, evidence clips, match stats dashboard, publish + share tokens |

Event inference is **heuristic** (not a trained event model) so the review workflow is complete; lock quality after footage eval.

## Quick start

```bash
export DOCKER_HOST=unix:///var/run/docker.sock   # if needed
docker compose up --build
```

- App: http://localhost:5173  
- API: http://localhost:8000/docs  

Env knobs: `VISION_BACKEND=yolo|stub`, `YOLO_MODEL=yolov5x_football.pt` (football player/GK/ref/ball; ~3–5× slower on CPU than `yolov8n.pt`), `VISION_DEVICE=cpu`, `VISION_FRAME_STRIDE=5`, `ALLOW_SINGLE_CAMERA=true`.

### Phase 0 CLI (no UI)

```bash
cd backend && source .venv/bin/activate
python -m app.phase0.benchmark /path/to/clip.mp4 --max-frames 200 --stride 5
```
