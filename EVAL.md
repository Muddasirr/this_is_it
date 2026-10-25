# CourtLens evaluation

**Rule:** finish one phase gate before starting the next. See `PRD.md` §10 and `ARCHITECTURE.md` §8.

**Lock (non-negotiable):** do not open possession / events / UI / Kaggle work while the current phase gate is red. Fix → run the gate command → only then advance.

## Court formats

| `COURT_FORMAT` | Field players / side | Typical on-court people |
| -------------- | -------------------- | ----------------------- |
| `5v5` | 5 | ~10–12 (+ GKs/refs) |
| `6v6` | 6 | ~12–14 |
| `7v7` | 7 | ~14–16 |

## Where to run

GPU work runs on **Kaggle** (T4) with the **same backend code** the product uses —
`app.scripts.run_offline` drives the real `process_job` on SQLite + local storage.
Local CPU is for unit tests, smoke runs (`--smoke`) and GPU-free re-inference.

```bash
# Build code bundle + kernel, upload, run (private dataset muddasirr/courtlens-code)
python scripts/kaggle/build_kaggle.py --push
kaggle kernels status muddasirr/courtlens-gpu-eval
kaggle kernels output muddasirr/courtlens-gpu-eval -p data/kaggle_out/gpu_eval
# → eval/eval_index.json + eval/<slug>/{summary,detections,events,canonical_events,players,match_stats}.json

# Same runner locally (1–4 cameras; needs ffmpeg)
cd backend && python -m app.scripts.run_offline --out ../data/run/x --camera A=a.mp4 --camera B=b.mp4

# Load a Kaggle run into the local app (one match per video; Run stats panel + facility table)
python scripts/import_kaggle_run.py data/kaggle_out/gpu_eval/eval [--only test_video_2,dfl_]

# Re-run identity → events → fuse → stats on saved detections, CPU, seconds
cd backend && python -m app.scripts.run_offline --replay ../data/kaggle_out/gpu_eval/eval/<slug> --out /tmp/re

# Pass precision/recall on labelled DFL clips (runs in the kernel; local needs the clips)
cd backend && python -m app.scripts.bench_events --clips <dfl-clips dir> --out bench.json --per-class 150

# Re-run events only on saved detections (no GPU) — event tuning loop
PYTHONPATH=backend python scripts/eval/reinfer_events.py data/kaggle_out/gpu_eval/eval/test_video_2 data/eval/reinfer/test_video_2
```

## Commands

```bash
# Unit + pipeline + API tests (incl. 2-camera end-to-end on SQLite with stub vision).
# Use the project venv — it has the pinned bcrypt/passlib versions.
cd backend && .venv/bin/python -m pytest -q

# Phase 0 — detection smoke (Roboflow or fixture)
python scripts/eval/run_detection_smoke.py

# Phase 0 — list corpus
python scripts/run_testing_videos.py --list

# Event P/R vs GT (Phase 4) — canonical events from a Kaggle run
python scripts/eval/score_events.py \
  --pred data/kaggle_out/gpu_eval/eval/test_video_2/canonical_events.json \
  --gt data/eval/gt/test_video_2_events.json

# Possession dump (Phase 3)
python scripts/eval/dump_possession.py \
  --detections data/kaggle_out/ocr_pull/courtlens_out/detections.json \
  --out data/eval/possession_dump.json
```

## Datasets

| Dataset | Use | Path |
| ------- | --- | ---- |
| Facility CCTV (`testing_videos/test_video_2.mp4`) | Product E2E (1 camera) | local + Kaggle `muddasirr/courtlens-test-video-2` |
| Highlights under `testing_videos/` | Stress only | local + same Kaggle dataset |
| DFL Bundesliga 30 s clips | Tracking / kit-split stress | Kaggle `saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv` (attached to the kernel; sample in `testing_videos/dfl_*`) |
| Roboflow Futsal-Object-Detection_v2 | Detection smoke | `data/eval/roboflow_futsal/` |
| Manual GT for `test_video_2` | Event P/R | `data/eval/gt/test_video_2_events.json` — **placeholder (3 fake events); must be hand-labelled** |
| Synthetic 3-camera set | Multi-cam sync / identity / fuse | generated in the Kaggle kernel from one clip (known offsets) |
| Futsal goal GT (no public labelled futsal set exists) | Goal recall | `backend/data/futsal_gt/`: Portugal–Spain 4 (video ends at 2-2, then recap), Spain–Brazil 1 (overlay frozen at 2-2 until the winner), Spain–France 7 (= 5-2 final). Bangladesh–Pakistan has no overlay → count only (6). `test_video_2` is a handheld phone clip; the wall LED is a clock, not the score, so its events still need hand labels in the app |
| **DFL labelled event clips** | **Pass precision / recall** | Kaggle `alejopaullier/dfl-clips`: ~2.5 s clips cut around the DFL competition labels — 2,849 `play` (pass), 598 `challenge`, 172 `throwin`, 1,802 `background`. `bench_events.py` → `bench_events.json` |

**Honest notes.** The DFL 460-clip dataset's `train.csv` is only a file list — **no event labels**
(train/test are inherited folder names). Event ground truth comes from `alejopaullier/dfl-clips`
instead (football broadcast — it scores the events engine, not futsal CCTV).
Until `test_video_2` GT is labelled by watching the match, every event precision/recall number
is meaningless — the Phase 4 gate below only checks "runs + no flood". There is **no real
multi-angle court footage** yet; multi-cam is validated synthetically only.

Download Roboflow (needs `ROBOFLOW_API_KEY`):

```bash
python scripts/eval/download_roboflow_futsal.py
```

Product default: `YOLO_MODEL=yolov8m.pt` (Kaggle v6: best player and ball recall on all 5 futsal videos). Previously `YOLO_MODEL=yolov5x_football.pt` (football_analysis YOLOv5x, classes `ball/goalkeeper/player/referee` remapped to CourtLens `sports_ball|person|referee`). Better referee and ball than COCO `yolov8n.pt`; ~3–5× slower on CPU. Facility CCTV with 0 players still auto-falls back to `yolov8n.pt`. Optional `yolov8n_futsal.pt` / `backend/models/futsal_yolov8n.pt` for Roboflow-like futsal still works the same way.

## Gates (pass/fail)

| Phase | Gate |
| ----- | ---- |
| 0 | Eval scripts run; GT template present; detection smoke exits 0 |
| 1 | `summary.json` has ball + person tracks on facility smoke |
| 2 | Kit/OCR unit tests (`pytest backend/tests/test_kit_identity.py backend/tests/test_jersey_ocr.py backend/tests/test_team_assigner.py`) + `python scripts/eval/build_identity_gallery.py data/kaggle_out/gpu_eval/eval` exits 0 (both kits on every run) + manual gallery confirm (one colour per kit letter). Tune with `run_offline --replay` — no GPU needed |
| 3 | Possession timeline non-empty when ball samples ≥ 80 |
| 4 | No flood (>40 shots on test_video_2) + `bench_events.json` pass precision/recall reported (play vs background/challenge/throw-in) + `bench_soccertrack_*.json` (full half). Futsal P/R: label a match in the app (Match centre → accept/reject + *Add missed event*), *Export ground truth*, score with `score_events.py --gt <export>` |
| 5 | `run_product_smoke.py` + share/UI per-player heatmaps; docker when available; Kaggle `multicam_synthetic` row `pass: true` (sync error ≤ 200 ms for B and C, ≥ 50 % players linked across cameras, ≥ 50 % events fused) |

## Heatmaps

Per-player grids live on `stats.by_player[].heatmap` and `possession_heatmap` (share report too). Match aggregates on `stats.heatmap.match` and `match_possession`. Always image-plane / `estimated`.

## Measured results (Kaggle v11, 2026-10-23)

| Feature | Ground truth | Result |
|---|---|---|
| Player detection | SoccerTrack v2 MOT (4096 px panorama, 2 clips) | recall 0.84 / 0.78 at auto 1920 px input (0.16 / 0.13 at the old 640) |
| Tracking | same | IDF1 0.46 / 0.48, ~155 ID switches per 600 frames — still the weakest link |
| Team split | same + 12 hand-labelled futsal CCTV players | 87 % / 85 % (SoccerTrack), 11/12 (CCTV) — torso colour + 0.25 × SigLIP |
| Team split + bystander filter (2026-10-24) | 60 hand-labelled tracks each on `test_video_2` and Spain–France (`bench_labels/*_roles.json`, `bench_teams.py`) | team accuracy **96 % / 100 %** (was 64 % / 52 %); spectators/coaches counted as players 0/20 and 2/24 (was 9/20, 9/24); possession 51/49 and 49/51 (was 76/24, 70/30). Kits decided per track from a zero-shot SigLIP colour profile, bystanders dropped by a zero-shot player prompt, players grouped within a kit (`role_filter.py`, `team_embed.track_kits`) |
| Passes | DFL labelled clips (400) | P 0.41 / R 0.66 / F1 0.50 |
| Passes | SoccerTrack M2, first 12 min, 167 labelled passes | P 0.38 / R 0.43 / F1 0.40 (ball-limited: ~8 px ball) |
| Shots / goals | SoccerTrack | 0 / 5 shots found — not yet reliable |
| Goals (futsal) | 12 broadcast-scoreboard goals (`backend/data/futsal_gt/*_goals.json`, `scoreboard_goals.py` + frame check) | Kaggle v9 predictions, `bench_goals.py` (window goal −15 s … +2 s): goal recall **1/12**, goal precision 1/5, a shot or goal before **3/12** goals. Edge-of-frame goal bands do not work on panning broadcast; needs calibrated goal mouths |
| Tracking + offline stitching (2026-10-24) | SoccerTrack MOT 117093 / 132877 (600 frames, ReID on, Modal L4) | IDF1 0.461 → **0.579** / 0.490 → **0.569**; ID switches 164 → 126 / 156 → 95 (`track_stitch.py`, `bench_mot --reid`). FP16 tried: +8–18 % fps but stitched IDF1 −0.02 → kept FP32 |
| Identity: roster slots (2026-10-24) | labelled phone clip + Spain–France | Kit X clusters no longer take kit roster slots (they pushed kit players into the other kit): team accuracy **100 % / 100 %** with stitching (was 64 % with stitching) |
| Jersey numbers, PARSeq (vendored, Apache-2.0) | SoccerNet GSR valid, 94 players (eval only) | 15 % of players correct, 30–48 % precision when it answers → auto numbers stay **off** |
| GPU processing (Modal) | 30 s DFL clip through `VISION_BACKEND=modal` | 143 s round trip incl. cold start; ~7.6 processed fps on L4 → a 40-min match ≈ 50 min (gate 30 min: needs profiling) |
| Calibration | 61 held-out SoccerTrack landmarks | 0.93 m mean error (8 points) vs 11 m (4 corners) |
| Jersey numbers | SoccerNet GSR valid, 94 players | trained classifier 11 %, Tesseract 7 % → auto numbers **off** (staff rename) |
| Multi-camera | synthetic 3-camera check | PASS (sync 0 ms, 15/16 players linked) |
| Product end to end | live app, 18 steps | PASS (CPU worker: 65 s of video in ~43 min; GPU needed for full matches) |

## Multi-camera (A–D)

`multicam_synthetic` in `eval_index.json`: B = cropped + brightened copy starting 3.0 s later
(expect `offset_ms ≈ +3000`), C = 1.25× zoom starting 1.5 s later (expect `≈ +1500`).
It checks the wiring of sync → per-camera identity → cross-camera link → fuse. It does **not**
prove quality on real angles (different viewpoints, occlusion); that needs a recorded 3–4 camera court match.

## Labels

All auto events stay `estimated` until staff confirm. Goals always review-friendly.
