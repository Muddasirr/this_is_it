# PadelLens — architecture

Post-match padel analytics from one fixed camera (club camera behind the back glass, or broadcast main view). Upload a match → calibrate the court once per camera → automatic players / ball / hits / strokes / rallies → per-player and per-pair stats → review and share.

Independent of the futsal/football product (`../backend`, `../frontend`): own package, own DB, own models, own Kaggle account. Nothing under `padel/` imports from the futsal code and vice versa.

---

## 1. Principles

1. **Exploit padel structure.** 4 players, 2 per side, a 10 × 20 m floor with strong lines, a fixed camera. Constrained trackers beat generic ones here.
2. **Every number has a provenance label** — `measured` (metres from calibrated tracks), `estimated` (model/heuristic: stroke type, volley, ball ground speed), `confirmed` (reviewed by a person).
3. **Held-out evaluation only.** Every model is scored on a match it never saw (PadelTracker100 cross-match). See `EVAL.md`.
4. **GPU for perception, CPU for everything else.** Detectors train/run on Kaggle (account `hadivenix`); trajectory, events, stats run anywhere on cached detections.
5. **Swappable stages.** Each stage is a pure function over arrays (`padel_analyzer/engine/*`) so a better ball model (WASB/TrackNetV3) or an audio hit cue drops in without touching the API.

## 2. Repository layout

```
padel/
  docs/            RESEARCH.md  ARCHITECTURE.md  ROADMAP.md  EVAL.md
  padel_analyzer/
    engine/        court.py  detect.py  players.py  ball.py  events.py  shots.py  rallies.py  stats.py  pipeline.py
    eval/          pt100.py (GT loaders)  metrics.py  run_eval.py
    api/           main.py (FastAPI)  db.py  worker.py  settings.py
  web/             Vite + React + TS single-page app
  kaggle/          kaggle_cli.py (second account)  prep/padel_prep.py (CPU)  gpu/padel_gpu.py (T4×2)
  scripts/         train_events.py  fetch_demo.py
  models/          ball.pt  pose.pt  hit_lgbm.txt  shot_lgbm.txt   (gitignored, pulled from Kaggle output)
  tests/
  data/            gitignored: PadelTracker100, Kaggle outputs, uploaded matches
```

## 3. Processing pipeline

```
video ──► ingest (probe fps/size, thumbnail, proxy frames for the UI)
      ──► court calibration  (UI: click ≥4 of 13 floor landmarks → H; optional auto-refine on white lines)
      ──► perception  per frame (GPU if present, else CPU with stride)
            · ball detector  YOLO11s @1536, top-5 candidates, conf ≥ 0.03
            · player pose    YOLO11m-pose @1280: boxes + 17 keypoints
      ──► players   court filter (feet inside the floor ± margin, in metres)
                    → 4-slot assignment: Hungarian on metres, constant-velocity prediction,
                      2 slots per net side; slots re-linked by torso colour after cuts / side changes
                    → positions smoothed (median 5 + Savitzky–Golay), speed clipped at 8 m/s
      ──► ball      Viterbi over candidates (cost: detector confidence + motion consistency + jump limit)
                    → gaps ≤ 6 frames interpolated, isolated blips dropped → px track + visibility
      ──► hits      per-frame features (ball velocity change, ball–wrist distance / player height,
                    wrist speed, ball side) → gradient-boosted classifier → peak picking (NMS 10 frames)
                    → hitter = closest wrist; heuristic fallback when no model file
      ──► strokes   RGB clip model on the hitter crop (R(2+1)D-18, sees the racket) fused 0.7/0.3 with a
                    pose/context GBM → {serve, forehand, backhand, smash, other}
                    + tags: volley, overhead, after-glass (estimated)
      ──► rallies   hits grouped (gap ≤ 3.5 s), serve opens a rally; last hitter; duration; length
      ──► stats     per player / pair / match + tactical (engine/advanced.py: serves, returns, zones,
                    direction, depth, net approaches, formation, work rate) + heatmaps + 5 Hz replay track
      ──► report.json  (+ review overrides stored separately, merged at read time)
```

### Coordinates

Court frame in metres: `x ∈ [0, 10]` across (camera left → right), `y ∈ [0, 20]` along, `y = 0` far back wall, net at `y = 10`, service lines at `y = 3.05` and `16.95`, centre service line `x = 5`. Same frame as PadelTracker100, so its homographies and floor positions are directly comparable.

Calibration landmarks (floor plane): `far_left (0,0)`, `far_right (10,0)`, `far_service_left (0,3.05)`, `far_service_t (5,3.05)`, `far_service_right (10,3.05)`, `net_left (0,10)`, `net_centre (5,10)`, `net_right (10,10)`, `near_service_left (0,16.95)`, `near_service_t (5,16.95)`, `near_service_right (10,16.95)`, `near_left (0,20)`, `near_right (10,20)`.

### Zones

- **Net zone:** within 4 m of the net (`|y − 10| ≤ 4`).
- **Transition:** 4 m to the service line (`4 < |y − 10| ≤ 6.95`).
- **Back zone:** behind the service line.

## 4. Stats (v1)

| Level | Stat | Label |
| --- | --- | --- |
| Player | distance (m), avg / max speed (km/h), sprints (> 4 m/s for ≥ 0.5 s) | measured |
| Player | time in net / transition / back zone (%), heatmap 0.5 m grid | measured |
| Player | shots by stroke type, volleys, overheads, after-glass shots | estimated |
| Player | avg ground speed of own shots (km/h) | estimated |
| Pair | % time both at net, avg partner distance, lateral spacing, side coverage | measured |
| Pair | shots, share of rally hits, rallies won (if confirmed) | estimated / confirmed |
| Match | rallies, avg / max rally length (shots), avg rally duration, shot-type mix, total hits | estimated |

## 5. Product surface

**API** (FastAPI, `padel_analyzer/api/main.py`, SQLite at `data/padel.db`, files in `data/matches/<id>/`):

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/matches` | upload video (multipart) → match |
| GET | `/api/matches`, `/api/matches/{id}` | list / detail (status, progress, calibration, player names) |
| GET | `/api/matches/{id}/video` | stream (HTTP range) |
| GET | `/api/matches/{id}/frame?t=` | JPEG still for calibration |
| PUT | `/api/matches/{id}/calibration` | landmark clicks → homography + reprojection error |
| POST | `/api/matches/{id}/calibration/auto` | line-based refinement / template init |
| POST | `/api/matches/{id}/analyze` | queue analysis (background worker) |
| GET | `/api/matches/{id}/report` | full report (stats, rallies, shots, heatmaps, replay track) |
| PATCH | `/api/matches/{id}/review` | player names, shot-type fixes, rally winners (confirmed) |
| GET/POST | `/api/cameras` | saved camera calibrations reusable across matches |

**Worker:** in-process thread pool (one analysis at a time on CPU; `PADEL_DEVICE=cuda` uses the GPU). Progress reported per stage.

**Web** (`web/`, Vite + React + TS): Matches (upload, status) → Calibrate (click landmarks on a still, live court overlay, error readout, reuse saved camera) → Report (video with synchronised mini-court replay, rally timeline, shot list that seeks the video, player & pair cards, heatmaps, review edits).

## 6. Compute

- **Kaggle (`hadivenix`)**: CPU kernel `padel-pt100-prep` downloads PadelTracker100 (parallel range requests, no GPU quota) and builds YOLO sets; GPU kernel `padel-gpu-train` (T4×2) trains cross-match + production detectors and caches held-out detections for every frame of both finals. `kaggle/kaggle_cli.py` pins every call to this account.
- **Kaggle only for compute.** `padel-gpu-pose` also trains the hit/stroke classifiers and produces every EVAL number on the held-out caches; `padel-stroke-rgb` trains the RGB clip stroke model; `padel-infer` analyses uploaded matches (API `PADEL_BACKEND=kaggle`: video + calibration → `padel-jobs` dataset version → kernel → `report.json`). Weights are never copied around: kernels mount the training kernels' outputs.
- **Local machine**: API + web app, file transfer to/from Kaggle. `PADEL_BACKEND=local` exists for a machine with its own GPU.
