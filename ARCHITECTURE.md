# CourtLens architecture — production futsal player analysis

**Product requirements:** `PRD.md`  
**Pilot constraints:** `PILOT.md`  
**Code:** `backend/app/` (API, workers, vision)

This document is the **technical source of truth** for how CourtLens produces player and match analytics in production: tracking → identity → possession events → stats → review → publish.

---

## 1. Design principles (production)

1. **League logic, facility cameras** — Same spine as pro leagues (dense tracks → possession → events → identity → human QA). Capture is 1–4 fixed CCTV angles (A–D) or uploaded match video, not a 16–20 camera optical install.
2. **Canonical truth** — Published stats come only from **canonical events** + track metrics linked through **identity**. Never count raw detections as player stats.
3. **Labels over false precision** — Every metric is `estimated` | `confirmed` | `unavailable`. Wrong identity is worse than a missing number.
4. **Staff override wins** — Model proposes; facility staff confirm identity and events before high-stakes publish.
5. **Swappable vision** — `VisionBackend.detect_and_track` / `infer_events` isolate model upgrades from product APIs.
6. **Reprocess = new run** — Model/config changes create a new `processing_run`; staff-confirmed work is not silently erased.
7. **Cost-aware compute** — On-demand GPU for detect/track; CPU for media/clips; configurable frame stride.

---

## 2. How top leagues do it (reference)

| League | Tracking | Events | Identity |
| ------ | -------- | ------ | -------- |
| Premier League | Second Spectrum optical (~25 Hz) | Opta / Stats Perform fused with tracking | Lineups + multi-camera |
| La Liga / Serie A / Ligue 1 | Optical / broadcast-augmented | Opta-style eventing | Lineups + jersey + multi-view |
| Bundesliga | 16–20 cameras, ~40 ms samples | ~1,600 events/match with **human operators** | Multi-camera + QA room |

Shared pattern:

1. Dense player + ball tracking.  
2. Events built on **possession** (who has the ball, when it changes).  
3. Identity from squads, kits, numbers, multi-view continuity.  
4. Human QA for publishable truth.  
5. Player reports **derived from events**, not detections alone.

Research baseline (e.g. arXiv:2202.00804): possession zones → gain/loss → classify pass/shot/clearance using geometry and kinematics.

CourtLens copies this **pipeline shape**, not the stadium hardware.

---

## 3. What CourtLens implements vs leagues

| League capability | CourtLens production v1 | Later (P1+) |
| ----------------- | ----------------------- | ----------- |
| Calibrated pitch metres | Image-plane units + relative distance; optional homography | Full court calibration |
| Multi-view re-ID | Per-camera clusters linked across angles by kit + shirt + OSNet (one-to-one per camera); staff merge | Learned cross-view re-ID |
| Jersey OCR | **In pipeline** (Tesseract when available); kit colour always | Stronger OCR / trained reader |
| Kit / team | Torso Lab k-means clustering → Kit A / Kit B (refs/outliers → Kit X) | Facility-trained kit models |
| Event ops room | Facility review UI | Optional ops tooling |
| Event taxonomy | Pass, shot, goal, assist, dribble | Saves, tackles, set pieces |
| Live | Deferred | Optional |

---

## 4. End-to-end processing architecture

```
Upload Camera A [(optional) Cameras B, C, D]
    → Ingest / validate (FFmpeg probe)
    → Normalize (proxy for review; store original)
    → Synchronize (offset per camera onto A: audio onset xcorr → motion-energy xcorr → staff fix)
    → Detect & track (YOLO + BoT-SORT; persons + sports_ball)
         · dense bbox_series (ts_ms, xyxy, frame_w/h, conf)
         · ball kept even without tracker ID (synthetic ball track)
         · secondary ball `predict` at conf≤0.12 / imgsz≥1280 when track miss
         · OSNet ReID embeddings (EMA per track) when VISION_REID=1
         · upper-body HSV colour_hist (fragment merge)
         · jersey OCR samples → majority number
         · best in-match thumb crop per track
    → Resolve identity
         · merge fragments by ReID cosine, then colour
         · assign Kit A / Kit B by torso Lab k-means (refs / outliers → Kit X)
         · fuse OCR shirt numbers (unique within kit)
         · create/update roster → TrackIdentityLink
         · staff rename / reassign in UI
    → Infer events (possession-centric)
         · pass / shot / goal candidates + roles
    → Fuse (same type within 1.5 s on the synced timeline, ≤1 obs per camera, players must agree)
    → Stats (match_stats from canonical events + tracks)
    → Review gate (clips; estimated → confirmed)
    → Publish (dashboard, player reports, share token)
```

### Production runtime topology

| Layer | Implementation |
| ----- | -------------- |
| Web | React + TypeScript (facility dashboard, review, share report) |
| API | FastAPI, facility-scoped auth |
| DB | PostgreSQL (metadata, events, identity, audit, jobs) |
| Queue | Redis + durable job rows |
| Storage | Local (dev) or private S3-compatible (prod) |
| Workers | `app.workers.runner` — media (CPU) + vision (GPU when `VISION_DEVICE=0`) |
| Detection | Ultralytics YOLO (`yolov5x_football.pt` default, `yolov8n.pt` fallback) + BoT-SORT (default, `VISION_TRACKER`) / ByteTrack; OSNet ReID merge (`yolo_backend.py`, `reid.py`) |
| Events | `events.py` possession heuristics |
| Identity | `identity.py` kit + jersey clustering |
| Stats | `match_stats.py` |
| Offline GPU batch | `app.scripts.run_offline` — the real `process_job` on SQLite + local storage; run on Kaggle T4 via `scripts/kaggle/build_kaggle.py` for eval / overflow batch — **not** the multi-tenant API path |

### Env knobs (vision)

| Variable | Role | Production default |
| -------- | ---- | ------------------ |
| `VISION_BACKEND` | `yolo` \| `stub` | `yolo` |
| `YOLO_MODEL` | Weights path | `yolov5x_football.pt` (football player/GK/ref/ball; resolves to `data/eval/football_analysis_model/best.pt`). Auto-falls back to `yolov8n.pt` if no persons (facility CCTV). `yolov8n_futsal.pt` remaps `player/gk/ball`. ~3–5× slower on CPU than `yolov8n.pt` |
| `VISION_DEVICE` | `cpu` \| `0` | `0` on GPU workers |
| `VISION_FRAME_STRIDE` | Process every Nth frame | `5` (balance quality/cost; raise on CPU) |
| `VISION_CONF` / `VISION_IMGSZ` | Detector thresholds | `0.25` / `640` |
| `COURT_FORMAT` | `5v5` \| `6v6` \| `7v7` | `5v5` |
| `ALLOW_SINGLE_CAMERA` | 1-cam matches | `true` |

---

## 5. Statistic definitions (locked)

Must stay aligned with `backend/app/services/events.py` and the review UI.

### Pass (`estimated` until confirmed)

1. Player A has **sustained possession** (ball nearest within possession radius for ≥ N samples).  
2. Ball leaves A’s zone (flight / gap).  
3. Player **B ≠ A** gains sustained possession within a time window.  
4. Transition is not classified as a shot.

**Passer** → role `passer`. Receiver → `receiver` (assist path).

### Shot (`estimated`)

Following Vidal-Codina et al. (arXiv:2202.00804) plus ISVP.AI futsal shot rules (icSPORTS 2023):
possession release that is **goalward + high ball speed/accel**, evaluated on a **fixed post-release window**,
and **not** followed by another field player gaining possession unless the ball reaches the goal band
(on-target / save / goal). Soft “kick motion” / sparse-ball shot floods are **disabled** when ball coverage is usable.
Goals require a prior shot + ball linger in the goal band (staff review still required).

**Shooter** → role `shooter`.

### Goal (`estimated`, always review-friendly)

Only after a detected shot, if the ball **lingers** in the extreme goal band (≥2 samples). Always confirm in review — CCTV geometry is not a calibrated goal line.

### Track-derived player metrics

| Metric | Source |
| ------ | ------ |
| Distance | Path length of linked person tracks (relative units; metres only after homography) |
| Heatmap | Binned foot/centre positions on image plane |
| Possession % | Blend of ball-holder time + on-camera presence (weights follow ball coverage) |
| Touches | Possession holder changes (ball timeline); if no ball samples, pass/shot/goal primary roles |
| Minutes | On-camera time from linked person tracks |
| Avg speed | Relative distance / minutes (image plane; not km/h) |
| Assist | Prior completed pass into a goal in the possession chain |
| Dribble | Progressive carry while the same holder keeps the ball |
| Pass received | Receiver role on completed passes |
| Conversion % | Goals / shots × 100 (None if no shots) |
| On-ball heatmap | Player location only while holding the ball (Vidal-Codina §3.2.1) |
| Team totals | Sum of player stats by `team_id` |

### Aggregation

```
by_player[player_id].pass  += pass events (role passer|primary), not rejected
by_player[player_id].shot  += shot events (role shooter|primary)
by_player[player_id].goal  += goal events (role scorer|primary)
by_player[player_id].assist += assist events (role assist|passer|primary)
by_player[player_id].dribble += dribble events (role carrier|dribbler|primary)
by_player[player_id].pass_received += pass events (role receiver)
```

Rejected events never count. UI shows confirmation state per metric.

---

## 6. Identity architecture (production)

Same-shirt numbers on opposite kits are allowed (**Kit A #10 ≠ Kit B #10**).

1. **BoT-SORT** (default; ByteTrack optional) short-term continuity per camera, plus OSNet ReID to merge fragments.  
2. **Colour:** mean L2-normalised HSV hist of upper-body crops (fragment merge only).  
3. **Cluster:** merge non-overlapping tracks with cosine colour similarity ≥ threshold.  
4. **Kit split:** per-player torso jersey colour (grass/skin/white masked, Lab k-means), then match-level k-means (k≤4): two largest distinct clusters → Kit A / Kit B, small clusters (refs, GKs, bibs) → Kit X (`team_assigner.py`; HSV fallback in `kit_color.py`).  
5. **Jersey:** OCR votes → `prefer_jersey_number` (2-digit preference; anti “10→1”).  
6. **Roster:** unique shirt within kit; auto-create `Kit A #N` / `Kit B #N` if needed.  
7. **Avatars:** best in-match crop uploaded for player card.  
8. **Staff:** rename, reassign track→player; confirmed links beat model proposals.

**Multi-camera (A–D).** The "two tracks at the same time ⇒ two people" rule only holds inside one camera, so:

- Steps 1–3 run **per camera**; the auto-player cap applies per camera.
- Step 4 (kits) runs **jointly** over every camera's clusters so Kit A is the same team on every angle.
- **Cross-camera link** (`link_clusters_across_cameras`): the camera with the most on-camera time is the reference; every other camera's clusters are matched one-to-one (Hungarian) to reference players. Same kit is required; readable different shirts are never linked; same kit + same shirt wins; otherwise OSNet cosine ≥ 0.6 (colour fallback). Unmatched clusters stay separate players for staff to merge.

---

## 7. Data model (core)

`facilities`, `facility_users`, `teams`, `players`, `match_rosters`  
`matches`, `camera_sources` (A–D), `video_assets`, `camera_syncs` (one offset per camera B–D, latest wins, manual beats auto)  
`processing_jobs`, `processing_runs`, `model_versions`  
`player_tracks` (+ `features` JSON: class, bbox_series, color_hist, jersey_*, thumb_*)  
`track_identity_links`  
`match_events`, `event_observations`, `event_participants`  
`event_reviews`, `event_corrections`, `highlight_clips`  
`share_tokens`, `usage_records`, `retention_policies`

**Rules**

- Canonical match timeline; keep per-camera timestamps + offset.  
- Track ID scoped to camera + run.  
- One real-world event → one canonical event.  
- Stats traceable to events + corrections.  
- Signed expiring URLs; facility ACL on every media/API access.

---

## 8. Phased pipeline + evaluation

Delivery order (hard gates in `EVAL.md` / `PRD.md` §10):

0. Eval harness + datasets  
1. Detect/track  
2. Identity (kit, OCR, avatar)  
3. Possession module (`possession.py`)  
4. Events + stats including **per-player heatmap**  
5. Product review/share + multi-cam (sync, cross-camera identity, fuse for A–D)

### Possession contract (Phase 3+)

- PZ = circle in image plane (`POSSESSION_RADIUS_FRAC * frame_diag`).  
- Gain: prefer ball Δspeed / Δdirection while in PZ (Vidal-Codina §4.2).  
- Loss: leave PZ with short re-entry grace.  
- Events consume possession segments only — never raw motion bursts when ball coverage is usable.

### Heatmaps

- Grid `HEAT_W × HEAT_H` from person `bbox_series` foot/centre (image plane).  
- Stored on each `by_player[].heatmap` and `heatmap.match`.  
- Share + match UI must allow **per-player** selection (not match-only).  
- Label: relative camera frame until homography (Phase 6).

### Multi-cam sync + fuse

- `CameraLabel` A | B | C | D. Camera A is the reference clock.  
- Sync: `canonical_ts = camera_ts + offset_ms[camera]` (`media.estimate_offset_ms`): audio-onset cross-correlation when both files have audio, else motion-energy cross-correlation; confidence < 0.3 ⇒ staff must check. Manual offsets per camera always win.  
- Fuse (`fuse.group_event_candidates`): same `event_type`, within 1.5 s on the canonical timeline, at most one observation per camera, and resolved players must not disagree. Raw tracker IDs are never compared across cameras. Each canonical event keeps every camera's `event_observation`; evidence clips come from the highest-confidence camera.  
- Pilot runs **A only**; production courts target 3–4 angles.

### Eval corpus

- Product: `testing_videos/test_video_2.mp4`.  
- Stress: other `testing_videos/*` (not product UX claims).  
- Detection: Roboflow futsal set via `scripts/eval/download_roboflow_futsal.py`.  
- DFL Bundesliga 30 s clips (Kaggle `saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv`): tracking / kit-split stress. **No event labels** (the CSV is a file list; the original competition labels are no longer downloadable).  
- Multi-cam: synthetic 3-camera check on Kaggle (shifted + cropped copies, known offsets) until real multi-angle court footage exists.  
- GPU runs: Kaggle kernel `muddasirr/courtlens-gpu-eval` runs the real backend. Commands: `EVAL.md`.

---

## 9. Known limits (honest, production)

| Limit | Product response |
| ----- | ---------------- |
| Sparse ball on wide CCTV | Surface ball coverage; keep shots/goals estimated; improve camera or model |
| Image-plane ≠ metres | Label distance as relative until homography |
| OCR fails on blurry kits | Kit clustering + staff rename |
| Heuristic events ≠ Opta | Review gate; never sell as official league feed |
| CPU-only workers | Slow; production uses GPU workers + documented stride |
| Kaggle/Colab GPU | Eval / batch overflow only (same code via `run_offline`) — not multi-tenant SaaS hosting |
| No real multi-angle footage yet | Multi-cam validated synthetically only; record a 3–4 camera pilot match before quality claims |

---

## 10. Evolution path

| Stage | Upgrade |
| ----- | ------- |
| **Now (prod v1)** | YOLO + BoT-SORT + ReID, kit+OCR identity, possession events, review, share, GPU workers |
| **Next** | Court homography; stronger ball detector / SAHI; eval harness + auto-accept gates |
| **Later** | Trained event model on facility annotations; cross-camera re-ID; SSO/exports |
| **Optional** | Broadcast-grade providers if venues install optical arrays |

---

## 11. Code map

| Concern | Module |
| ------- | ------ |
| Detect / track / colour / OCR / thumbs | `backend/app/services/yolo_backend.py` |
| Jersey OCR helpers | `backend/app/services/jersey_ocr.py` |
| Possession (PZ / gain-loss) | `backend/app/services/possession.py` |
| Possession → pass/shot/goal | `backend/app/services/events.py` |
| Cross-cam fuse A–D | `backend/app/services/fuse.py` |
| Camera sync (audio / motion xcorr) | `backend/app/services/media.py` |
| Cross-camera identity link | `backend/app/services/identity.py` (`link_clusters_across_cameras`) |
| Court format 5v5/6v6/7v7 | `backend/app/services/court_format.py` |
| Kit + roster identity | `backend/app/services/identity.py` |
| Avatars | `backend/app/services/avatars.py` |
| Match / player stats (+ per-player heatmap) | `backend/app/services/match_stats.py` |
| Job stages | `backend/app/workers/pipeline.py` |
| Stats + review API | `backend/app/api/review.py` |
| Offline / Kaggle run (real pipeline on SQLite) | `backend/app/scripts/run_offline.py`, `scripts/kaggle/build_kaggle.py` |
| Product requirements | `PRD.md` |

---

## 12. Alignment checklist (docs ↔ code)

When changing behaviour, update **both** code and this file:

- [ ] Event definitions still match `events.py`  
- [ ] Identity still Kit A/B + shirt uniqueness  
- [ ] Stats still ignore rejected events  
- [ ] Labels `estimated` / `confirmed` / `unavailable` still enforced in UI  
- [ ] `VisionBackend` remains the swap point for new models  
