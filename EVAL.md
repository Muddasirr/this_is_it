# CourtLens evaluation

**Rule:** finish one phase gate before starting the next. See `PRD.md` §10 and `ARCHITECTURE.md` §8.

**Lock (non-negotiable):** do not open possession / events / UI / Kaggle work while the current phase gate is red. Fix → run the gate command → only then advance.

## Court formats

| `COURT_FORMAT` | Field players / side | Typical on-court people |
| -------------- | -------------------- | ----------------------- |
| `5v5` | 5 | ~10–12 (+ GKs/refs) |
| `6v6` | 6 | ~12–14 |
| `7v7` | 7 | ~14–16 |

## Commands

```bash
# Phase 0 — detection smoke (Roboflow or fixture)
python scripts/eval/run_detection_smoke.py

# Phase 0 — list corpus
python scripts/run_testing_videos.py --list

# Event P/R vs GT (Phase 4)
python scripts/eval/score_events.py \
  --pred data/kaggle_out/ocr_reinfer/events.json \
  --gt scripts/eval/fixtures/test_video_2_events.json

# Possession dump (Phase 3)
python scripts/eval/dump_possession.py \
  --detections data/kaggle_out/ocr_pull/courtlens_out/detections.json \
  --out data/eval/possession_dump.json
```

## Datasets

| Dataset | Use | Path |
| ------- | --- | ---- |
| Facility CCTV (`testing_videos/test_video_2.mp4`) | Product E2E | local |
| Highlights under `testing_videos/` | Stress only | local |
| Roboflow Futsal-Object-Detection_v2 | Detection smoke | `data/eval/roboflow_futsal/` |
| Manual GT for `test_video_2` | Event P/R | `data/eval/gt/test_video_2_events.json` |

Download Roboflow (needs `ROBOFLOW_API_KEY`):

```bash
python scripts/eval/download_roboflow_futsal.py
```

Product default: `YOLO_MODEL=yolov5x_football.pt` (football_analysis YOLOv5x, classes `ball/goalkeeper/player/referee` remapped to CourtLens `sports_ball|person|referee`). Better referee and ball than COCO `yolov8n.pt`; ~3–5× slower on CPU. Facility CCTV with 0 players still auto-falls back to `yolov8n.pt`. Optional `yolov8n_futsal.pt` / `backend/models/futsal_yolov8n.pt` for Roboflow-like futsal still works the same way.

## Gates (pass/fail)

| Phase | Gate |
| ----- | ---- |
| 0 | Eval scripts run; GT template present; detection smoke exits 0 |
| 1 | `summary.json` has ball + person tracks on facility smoke |
| 2 | Kit A/B any-colour 2-means + jersey OCR 2-digit (`pytest backend/tests/test_kit_identity.py backend/tests/test_jersey_ocr.py`) + `python scripts/eval/phase2_identity_loop.py` → `data/eval/identity_gallery/PHASE2_REPORT.json` + manual gallery confirm |
| 3 | Possession timeline non-empty when ball samples ≥ 80 |
| 4 | `score_events.py` runs; no flood (>40 shots on test_video_2) |
| 5 | `run_product_smoke.py` + share/UI per-player heatmaps; docker when available |

## Heatmaps

Per-player grids live on `stats.by_player[].heatmap` and `possession_heatmap` (share report too). Match aggregates on `stats.heatmap.match` and `match_possession`. Always image-plane / `estimated`.

## Labels

All auto events stay `estimated` until staff confirm. Goals always review-friendly.
