# PadelLens — padel match analytics

Post-match analytics for padel from one fixed camera: upload a match, mark the court once, and get per-player and per-pair movement, shots by stroke type, rallies, heatmaps, and a synchronised replay, with a review workflow to confirm what the models estimate.

Independent of the futsal product in the parent repo: separate package, database, models and Kaggle account. Nothing here imports from `../backend` or `../frontend`.

Docs: [docs/RESEARCH.md](docs/RESEARCH.md) (methods, datasets, prior art) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (pipeline, stats, API) · [docs/ROADMAP.md](docs/ROADMAP.md) (features and gates) · [docs/EVAL.md](docs/EVAL.md) (measured results).

## Deploy (production)

```bash
cd padel
export KAGGLE_API_KEY_LSECOND=...      # Kaggle account that runs the GPU analysis jobs
export PADEL_API_TOKEN=$(openssl rand -hex 16)   # optional: require an access token
docker compose up -d --build           # -> http://<host>:8010  (health: /api/health)
```

The container serves the web app + API and stores the database and uploaded matches in the `padel_data` volume. Every analysis runs on the Kaggle GPU (`padel-infer` kernel, models mounted from the training kernels): upload → `padel-jobs` dataset → GPU → `report.json`. A 25-minute match takes roughly 40–60 min end to end, mostly upload + GPU time. Jobs interrupted by a restart are marked failed and can be re-run from the UI; Kaggle calls retry with backoff and GPU jobs stop after `PADEL_KAGGLE_TIMEOUT_S` (default 9 h; Kaggle's own limit is 12 h).

Put it behind HTTPS (any reverse proxy) when exposed; set `PADEL_CORS` to the public origin.

Exports per match: `/api/matches/{id}/export/{shots|players|pairs|rallies}.csv` and the full report JSON (buttons on the report page).

## Run it (development)

```bash
cd padel
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
(cd web && npm install --legacy-peer-deps && npm run build)
.venv/bin/uvicorn padel_analyzer.api.main:app --port 8010      # app + API on http://localhost:8010
```

For frontend development, run `cd web && npm run dev` (http://localhost:5175, proxies `/api` to :8010).

The flow is: **Matches** (upload) → **Calibrate** (click 4+ court landmarks; save the camera to reuse it) → **Analyse** → **Report** (video with overlay, live mini-court, rally timeline, players / pairs / rallies / shots, review edits).

Environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `PADEL_BACKEND` | `kaggle` | `kaggle`: each analysis uploads the video + calibration as the `padel-jobs` dataset and runs the `padel-infer` GPU kernel on the padel Kaggle account, this machine only moves files; `local`: run on this machine |
| `PADEL_DEVICE` | auto | `cuda` or `cpu` (local backend) |
| `PADEL_POSE_STRIDE` | 2 | run player pose every n frames |
| `PADEL_MAX_FRAMES` | 0 | cap frames per analysis (0 = all) |
| `PADEL_DATA_DIR` | `data/app` | DB and uploaded matches |
| `PADEL_API_TOKEN` | – | when set, every `/api` call needs `Authorization: Bearer <token>` (the web app asks once) |
| `PADEL_KAGGLE_TIMEOUT_S` | 32400 | stop waiting for a Kaggle GPU job after this many seconds |
| `PADEL_CORS` | localhost:5175 | allowed browser origins |

Models (trained on Kaggle; `models/` is only needed for `PADEL_BACKEND=local`):
- `ball.pt` (ball_v2: PT100 + Roboflow venues), `pose.pt` (players + keypoints)
- `audio_hit.pt` (racket-impact sound CRNN) + `hit_lgbm.txt` (audio + video hit model, modality dropout)
- `stroke_rgb.pt` (clip model on the hitter crop) + `shot_lgbm.txt` (pose / context stroke model)

Without trained detectors the app falls back to stock YOLO11 (COCO sports-ball + yolo11m-pose); without the classifiers it uses rule-based scores.

## Data and training

- **PadelTracker100** (CC BY 4.0) is the ground truth. Fetch the labels locally with `kaggle/prep/padel_prep.py --out data/pt100 --skip-frames`.
- **Kaggle (second account, `KAGGLE_API_KEY_LSECOND` in `../.env`)**: run every call through `kaggle/kaggle_cli.py` so it never touches the futsal account.
  - `python kaggle/kaggle_cli.py push prep` runs a CPU kernel that downloads both finals and builds YOLO sets.
  - `python kaggle/kaggle_cli.py push gpu` runs on T4×2: it trains the cross-match and production ball and pose detectors, and caches held-out detections.
  - `python kaggle/kaggle_cli.py output gpu data/kaggle_out/gpu` downloads the results.
- `python kaggle/kaggle_cli.py code` uploads `padel_analyzer/` + `scripts/` as the `padel-code` dataset that every kernel mounts.
- `python kaggle/kaggle_cli.py push pose` trains the pose detectors, caches held-out detections, then runs `scripts/train_events.py --source heldout --save` and `python -m padel_analyzer.eval.run_eval` **on Kaggle** (hit/stroke models + every EVAL number).
- `python kaggle/kaggle_cli.py push stroke` trains the RGB clip stroke classifier (`stroke_rgb.pt`, sees the racket hand).
- `padel-infer` (pushed by the API worker) mounts all trained weights directly from the training kernels' outputs.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

The tests cover court geometry, calibration, the ball linker, the 4-player tracker and cuts, rally grouping, report shape, and the API flow (upload → calibrate → analyse → report → review).
