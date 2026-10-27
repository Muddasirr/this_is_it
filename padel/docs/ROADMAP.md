# PadelLens — roadmap

Feature by feature: implement → unit test → measure on held-out labelled data → record in `EVAL.md` → next. A feature is done only with a number.

Status: ✅ done (measured) · 🟡 in progress · ⬜ not started

## P0 — End-to-end v1 on PadelTracker100 (this sprint)

| # | Feature | Method | Gate (held-out match) | Status |
| --- | --- | --- | --- | --- |
| 0 | Data + compute | PT100 v1/v2 on Kaggle `hadivenix` (CPU prep kernel), cross-match YOLO sets | both finals + labels + frames on Kaggle | ✅ |
| 1 | Court model + calibration | 13 floor landmarks → H (RANSAC ≥5), reprojection error; line-based auto-refine | manual H vs PT100 H: ≤ 0.15 m mean floor error; snap with 6 px clicks: ≤ 0.25 m — **0.077 m (4 clicks) / 0.075 m (snap)** | ✅ |
| 2 | Player detection + pose | YOLO11m-pose fine-tuned cross-match @1280 | P/R ≥ 0.97 (IoU 0.5); floor position error ≤ 0.30 m median — **0.998/0.999 (F), 0.974/0.988 (M); 5 cm median** | ✅ |
| 3 | 4-player tracking + teams | constrained 4-slot Hungarian, team = net side, cut detection, kit-colour team linking | distance within 5 % of GT (oracle: ≤ 1.4 % per pair ✅); team linking 16/16 segments ✅; held-out pair distance −1.2…+1.7 % | ✅ |
| 4 | Ball | YOLO11s @1536 top-5 + trajectory linker (tuned cross-match) | recall@10 px ≥ 0.80, precision ≥ 0.85 (ref: HASHU7 88/89, lmrt 80) — **in-rally 0.913/0.873 (F), 0.913/0.896 (M)** | ✅ |
| 5 | Hit detection | GBM over trajectory + pose features, peak picking | F1 ≥ 0.85 at ±7 frames (ref: 0.88–0.91) — **held-out 0.880 (F) / 0.868 (M)** | ✅ |
| 6 | Stroke type | RGB clip model (R(2+1)D-18 on the hitter crop) fused 0.7/0.3 with pose GBM | accuracy ≥ 0.75 on GT hits (ref: 0.81) — **0.788 (F) / 0.783 (M)** | ✅ |
| 7 | Rallies + stats | grouping, zones, distances, pair stats, heatmaps | rally count within 10 % of GT-derived rallies — **+3 % (F) / −8 % (M)** | ✅ |
| 8 | Product | FastAPI + SQLite + worker; React app: upload → calibrate → report → review | upload → report on a real match video, all pages working — done on the PT100 sample clip through the real API + UI | ✅ |

## Shipped v1.0 (2026-10-27)

Docker image (API + web, analysis on the Kaggle GPU), optional access token, CSV/JSON exports, retrying Kaggle jobs with timeouts and restart recovery, audio + video hit detection, RGB stroke model, tactical stats (STATS.md). Elevated facility / broadcast cameras are the target view (all accuracy numbers in EVAL.md are for that view).

## P1 — Next

- **Racket-hand evidence for strokes** (biggest stroke-type lever, see EVAL §2): racket detector / racket keypoint, or an RGB clip classifier on the hitter crop (X3D-S / VideoMAE-S) trained on PT100 shot windows on Kaggle.

- Venue diversity: add Roboflow padel sets (ball/player) + 2–3 YouTube club-camera matches with hand-checked frames; retrain production detectors.
- Ball model upgrade if recall < 0.85: 3-frame heatmap model (WASB / TrackNetV3 with inpainting rectification), trained on PT100 at 512×288 crops.
- Auto calibration without clicks: court-keypoint model (YOLO-pose with 13 landmarks; labels generated from PT100 homographies + Roboflow keypoint set) + line refinement.
- ✅ Audio hit cue (Decorte et al. CRNN) fused with the visual features: F1 0.94 on 11 unseen tournaments (video-only 0.83) — EVAL §4.
- ❌ Padel shot-type detector from padelcnn/padel-quret: 0 % coverage on broadcast and club footage → not shipped; needs labelled match footage (label bandeja / víbora / lob in the review UI → train).
- Bounce detection (floor vs glass) with physics consistency; after-glass tag from measured rebounds.

## P2 — Padel-native insights

- Bandeja / víbora / smash separation (needs ~300 labelled overheads; label in the review UI → active learning).
- Lob detection and lob-defence success; net-recovery time after a lob.
- Serve & return: serve side, first/second serve, return depth.
- Points and score: scoreboard OCR on broadcast; rally-end classifier (winner / forced / unforced error) + review on club footage.
- Pair tactics: "both at net" %, parallel vs diagonal positioning, gap at the centre.
- Highlight reels (longest rallies, smashes) and shareable player pages.

## P3 — Production

- GPU worker (Modal/Kaggle-free path: on-demand T4), job queue, S3 storage, auth and club accounts — reuse patterns from the futsal product but no shared code.
- Live / near-live mode for club courts with a fixed camera.

## Risks

| Risk | Mitigation |
| --- | --- |
| Only one tournament / camera in labelled data | P1 venue diversity, Decorte set as a second test, held-out reporting per venue |
| Ball invisible against glass reflections / far court | temporal model, Viterbi path, gap interpolation, audio cue |
| Broadcast cuts and replays break tracks | scene-cut detection (frame-difference spikes) resets trackers; replays dropped |
| CPU inference too slow for full matches | stride for players, GPU worker for production |
