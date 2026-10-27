# CourtLens — roadmap to a production-grade, accurate app

What has to happen, in order, to go from today's working product shell to stats a
futsal facility can trust. Every step has a **data source**, a **measurement** and an
**exit gate**; nothing ships as "auto" until its gate passes on held-out footage.
Requirements live in `PRD.md`, stat definitions in `ARCHITECTURE.md`, how to run
the benchmarks in `EVAL.md`.

## 1. Where we are (measured 2026-10-23)

| Area | Today | Status |
|---|---|---|
| Upload → process → review → publish → share, roles, A–D cameras | 18/18 live end-to-end check | **Works** |
| Player detection | recall 0.84 / 0.78 (SoccerTrack) | Usable |
| Team split | 96 % phone clip / 100 % Spain–France (60 labelled tracks each); bystanders filtered | Usable — 2 videos measured |
| Court calibration (8 points) | 0.93 m mean error | Usable on fixed cameras |
| Tracking | IDF1 0.57–0.58 with offline stitching (was 0.46–0.49), ~95–126 ID switches / 600 frames | Improving — still below the 0.75 gate |
| Ball | false balls on flags / stands / boards | **Weak** — corrupts possession and passes |
| Passes | F1 0.50 (DFL clips), 0.40 (SoccerTrack) — hand-written rules, plateaued | **Weak** |
| Shots / goals | 0/5 shots (SoccerTrack); 1/12 futsal goals | **Not working** |
| Jersey numbers | 15 % (PARSeq, vendored) / 11 % (classifier) → auto numbers off, staff rename | **Not working** |
| Pretrained event model (T-DEED, SoccerNet) | pass F1 0.11, 0/12 goals — does not transfer | Ruled out as-is |
| Processing speed | Modal GPU path live (`VISION_BACKEND=modal`); ~50 min per 40-min match | Needs ~2× speed-up for the gate |
| Real fixed-camera futsal footage with labels | **none** | **Biggest gap** |

## 2. Target ("production gate") numbers

Locked per PRD §5.2 once the held-out futsal set (W0) exists. Proposed values:

| Metric | Gate | Measured on |
|---|---|---|
| Player detection recall / precision | ≥ 0.95 / ≥ 0.95 | venue frames (W0) |
| Ball detection recall (visible ball) | ≥ 0.80, false balls ≤ 5 % | venue frames |
| Tracking IDF1 (whole match, offline) | ≥ 0.75 | venue MOT clips |
| Team split accuracy | ≥ 97 % | venue tracks |
| Player identity correct for the whole match (after one staff naming pass) | ≥ 95 % of minutes | venue matches |
| Pass precision / recall (±1 s) | ≥ 0.80 / ≥ 0.70 | venue events |
| Pass attributed to the correct passer | ≥ 85 % | venue events |
| Dribble / carry F1 | ≥ 0.70 | venue events |
| Shot precision / recall | ≥ 0.80 / ≥ 0.70 | venue events |
| Goal recall (auto + one-tap staff tag) | 100 % before publish | review flow |
| Distance error (calibrated) | ≤ 5 % | venue match with a known run |
| Processing time, 40-min match, 1 camera | ≤ 30 min | GPU worker |
| Job success without an engineer | ≥ 99 % | production logs |

## 3. Player and match stats

Every stat carries a label on screen and on the share page: `confirmed` (staff-checked),
`estimated` (model output) or `unavailable` (not enough evidence). Status = today;
"Ships auto when" = the workstream gate that has to pass first.

### 3.1 Player stats (per player, per match)

| Stat | Definition | Built from | Status today | Ships auto when |
|---|---|---|---|---|
| Name / number / team | Who the player is | Kit split + jersey number + roster + staff rename | Team ok; numbers off; staff names | W4 + W5 |
| Minutes played | Time the player is on court | Stitched tracks | Rough (ID swaps split players) | W3 |
| Distance covered (m) | Foot-point path on the court, jitter-filtered | Tracks + calibration | Works on calibrated fixed cameras only | W3 + W6 |
| Average / top speed (km/h) | Mean speed on court; fastest 1 s window | Tracks + calibration | Same as distance | W3 + W6 |
| Heatmap | Where the player was on the court | Tracks (+ calibration for court coordinates) | Image-plane only | W3 + W6 |
| Possession % | Share of team ball-control time held by the player | Ball + possession zones | Estimated, hurt by false balls | W2 + W3 |
| Touches | Number of times the player gains the ball | Possession changes | Estimated | W2 + W3 |
| Passes made / received | Completed passes to / from a teammate | Event spotter + possession (passer, receiver) | Estimated (F1 0.40–0.50) | W7 |
| Pass accuracy % | Completed ÷ attempted (turnovers count as failed) | Passes + team check | Shown (estimated) | W7 |
| Interceptions / balls lost | Ball won from / given to the other team | Cross-team "passes" + staff tags | Shown (estimated) | W7 |
| Fouls / cards | Fouls committed, yellow / red cards | Staff tag | Staff tag | — (staff) |
| Keeper saves | Shots stopped by the goalkeeper | Staff tag | Staff tag | W7 |
| Dribbles / carries | Ball carried ≥ threshold distance while keeping possession | Spotter `DRIVE` + carry metres | Estimated | W7 |
| Shots / on target | Attempts on goal; on target = goal or save follows (or staff call) | Spotter `SHOT` (GPU) + saves/goals | Shots 3/7 found (GPU); on target derived | W7 + W6.2 |
| Goals | Ball over the goal line between the posts | Geometry + spotter + staff tag | Staff tag only | W7.4 (always confirmable) |
| Assists | Last completed pass to the scorer in the goal move | Goals + passes | Follows goals | W7.6 |
| Conversion % | Goals ÷ shots | Goals + shots | Follows shots | W7 |
| Possession heatmap | Where the player was while holding the ball | Possession + tracks | Estimated | W2 + W3 |

### 3.2 Match stats (per team and whole match)

| Stat | Definition | Built from | Status today | Ships auto when |
|---|---|---|---|---|
| Score + goal timeline | Goals with time, scorer, assist | Goals (W7) + staff confirmation | Staff tag | W7.4 |
| Possession % per team | Share of ball-control time | Possession + team split | Estimated | W2 + W4 |
| Passes / pass accuracy per team | Sum of player passes; completed ÷ attempted | Player passes | Estimated | W7 |
| Shots / on target per team | Sum of player shots | Player shots | Not working | W7 |
| Dribbles per team | Sum of player carries | Player dribbles | Estimated | W7 |
| Accumulated fouls per half | Team fouls; from the 6th in a half a 10 m kick | Staff-tagged fouls | Staff tag (half = 20 min of video) | — (staff) |
| Corners / kick-ins / free kicks / penalties | Set pieces per team | Staff tag | Staff tag | — (staff) |
| Passing combinations | Most frequent passer → receiver pairs | Passes | Estimated | W7 |
| Total distance per team (m) | Sum of player distances | Player distance | Calibrated cameras only | W3 + W6 |
| Team heatmap / match heatmap | Where each team played | Tracks | Image-plane | W6 |
| Key moments | Clips of goals, shots, big chances | Events + video | Clip export works; events weak | W7 |
| Data quality meters | Ball coverage %, tracking quality, calibration status | Pipeline diagnostics | Ball coverage shown | W8.2 |

## 4. Workstreams

### W0 — Ground-truth futsal data (blocks every accuracy claim)

1. **Record** 3–5 full matches at the pilot venue: fixed camera(s) covering the whole court
   (phone on a tripod is enough to start; production rig per PRD §8), 1080p, ≥ 25 fps.
2. **Write down goal times** during the match (paper or the app's review screen).
3. **Label events** for 2 matches in the app: Match centre → accept / reject / *Add missed
   event* → *Export ground truth* (pass, shot, goal, dribble, with player).
4. **Label boxes** for ~1,500 frames (players, ball, goalkeeper, referee) in CVAT or Roboflow,
   plus 3 × 1-min MOT clips (consistent IDs).
5. Split: 1 match = held-out test (never trained on), the rest = train / validation.
6. Keep collecting: every staff-reviewed match is exported to the training pool.

**Gate:** ≥ 1 held-out match with full event + box labels.

### W1 — GPU processing

0. **Done 2026-10-27 — Kaggle GPU worker path (current default):** `VISION_BACKEND=kaggle`
   (`backend/app/services/kaggle_backend.py`) uploads the video + the worker's own code as a
   temporary private dataset, runs detect + track + the futsal T-DEED spotter on a free T4
   (runner kernel `courtlens-job-runner`), downloads the result and deletes the upload. Network
   blips / Kaggle 429 / 5xx are retried for ~10 min; any other failure falls back to local CPU.
   Measured: 20 s clip round trip 345 s (mostly queue + setup). Quota 30 GPU-h/week.
1. **Done 2026-10-24, paused 2026-10-25 (credits) — Modal GPU worker path:** `VISION_BACKEND=modal` (worker default) sends
   detect + track to `courtlens.detect_video` on a Modal L4 (`scripts/modal/courtlens_gpu.py`,
   `modal deploy`); identity / events / stats stay on the worker. Falls back to local CPU YOLO
   when Modal is unavailable, so a job never fails on it. Kaggle stays the batch-eval source.
2. **Production:** same function scales to zero; track GPU seconds per match (PRD §6.5).
3. Batch frames and run ball tiles only where needed.

**Gate:** 40-min match ≤ 30 min, ≥ 99 % job success over 20 matches.

### W2 — Detection (players + ball)

1. Fine-tune `yolov8m`/YOLO11 on: W0 frames + Roboflow futsal sets + labelled football-ball
   sets (`michaelmortenson/football-soccer-ball-detection-dfl`,
   `tiagorodrigues26/ball-annotated-3k-yolov5-pytorch` on Kaggle) + SoccerTrack boxes.
2. Separate high-resolution ball model (tiles / SAHI) with hard negatives:
   corner flags, cones, boards, crowd, lights.
3. Ball on-pitch gate: reject balls outside the calibrated court polygon (fixed cameras).

**Gate:** W0 numbers in §2; DFL false-ball review on 50 clips < 5 %.

### W3 — Tracking and re-identification

1. Post-match → offline tracking: BoT-SORT with tuned buffers, then **global track stitching**
   (appearance + kit + court position + time gaps) across the whole match.
2. Fine-tune the ReID model on SoccerNet ReID / SoccerTrack crops + W0 crops.
3. Use court coordinates (calibration) for motion gating, not pixels.
4. Hard cap: 2 × court format + GK + refs active tracks after stitching.

**Gate:** IDF1 ≥ 0.75 on SoccerTrack and W0 MOT clips; ≤ 1 identity merge per player-minute.

### W4 — Team split and roles

Done 2026-10-24: bystander filter + per-track kits from a zero-shot colour profile + players grouped
within a kit (64 % / 52 % → 96 % / 100 % on two labelled videos, `bench_teams.py`).

1. Add goalkeeper / referee classes from the W2 detector (keepers currently join a kit by colour or stay unassigned).
2. Match-level clustering on stitched tracks (W3), not per short track.
3. Label more videos (SoccerTrack white/blue, a dark-vs-dark kit pair) before trusting it on every kit.

**Gate:** ≥ 97 % on W0 tracks.

### W5 — Player identity and jersey numbers

1. ~~`mkoshkina/jersey-number-pipeline`~~ is CC BY-NC 3.0 (no commercial use). Use its
   components that allow it: PARSeq scene-text recogniser (Apache-2.0) + our own legibility
   filter, voted over the stitched track.
2. Fine-tune on W0 / reviewed-match crops (commercially usable). SoccerNet jersey data is
   research-only → evaluation only, never weights we ship.
3. Roster import per match (names + numbers) so a read number maps to a person.
4. Staff naming pass in review stays mandatory until the gate passes.

**Gate:** number correct on ≥ 90 % of stitched tracks with a visible back; identity ≥ 95 % of minutes.

### W6 — Court calibration

1. Keep the 8-point manual calibration per fixed camera (one-time setup per court).
2. Store goal-mouth posts in calibration (needed by W7 goals).
3. Later: automatic futsal court keypoint model (train on W0 + SoccerNet calibration).

**Gate:** ≤ 0.5 m mean reprojection error on held-out court points; distance error ≤ 5 %.

### W7 — Events (the core of the product)

1. **Learned spotter:** fine-tune T-DEED on SoccerTrack v2 BAS labels (fixed whole-pitch camera,
   ~21k events) on Kaggle; test on held-out SoccerTrack match.
   Gate to continue: pass F1 clearly > 0.40 on SoccerTrack.
2. **Futsal fine-tune:** continue training on W0 + reviewed matches.
3. **Who:** passer = possession holder at release, receiver = next holder, from the W3 tracks;
   team check (a pass goes to a teammate; to an opponent = turnover).
4. **Goals:** ball crosses the goal line between the calibrated posts (W6) ∧ spotter `GOAL`/`SHOT`
   ∧ restart from centre; plus one-tap staff tag in review. Goals are never auto-published unconfirmed.
5. **Shots:** spotter `SHOT` + ball heading into the goal frame (geometry).
6. **Assists:** last completed pass to the scorer within the goal possession chain.
7. **Dribbles:** spotter `DRIVE` + carry distance in metres.
8. Retire the hand-written rules once the spotter beats them on W0.

**Gate:** §2 event numbers on the held-out futsal match.

### W8 — Stats, aggregation and trust UI

1. Per-player: minutes, distance (m), speeds, possession %, touches, passes made/received,
   pass accuracy, dribbles, shots, goals, assists, heatmaps — each with confirmed /
   estimated / unavailable.
2. Show ball-coverage and tracking-quality meters per match.
3. Hide metres/speeds on uncalibrated or moving cameras.
4. Review screen: one-tap goal tagging, bulk accept, jump to clip; measure review minutes/match.
5. **Done 2026-10-27 — futsal stat set:** staff tag fouls, yellow / red cards, keeper saves, own
   goals, corners, kick-ins, free kicks, penalties (and any missed pass / shot / goal); the model's
   "pass" to the other team is stored as an interception (won by / lost by). Derived: shots on
   target (goal or save ≤ 4 s after, or staff call), pass accuracy (completed ÷ completed +
   intercepted), conversion, score with own goals, accumulated fouls per half, passing pairs.
   Team stats are a side-by-side comparison table.
6. **Done 2026-10-27 — staff identity board:** every person track as a crop grouped by player;
   move a crop, merge a whole player, or mark crops "not a player" (bulk
   `PUT …/tracks/identity`). Per-player stats follow the links.

**Gate:** staff review ≤ 10 min per match; zero unlabelled estimates on share pages.

### W9 — Multi-camera (2–4 angles)

1. Record one real 2–4 camera match (W0).
2. Validate sync (audio / motion), cross-camera identity linking, event fusion on real footage.

**Gate:** sync error ≤ 100 ms; no double-counted events on the labelled match.

### W10 — Production hardening

1. Postgres + S3 storage, signed URLs, retention jobs (in place) → load test 50 concurrent uploads.
2. Monitoring: job durations, failure alerts, GPU cost per match, error tracking (Sentry).
3. Security: tenant isolation tests, auth hardening, secrets in a vault, dependency scanning.
4. CI: unit tests + `bench_*` regression suite on every change (fails if a metric drops).
5. Privacy: consent copy, deletion path, data processing agreement with the facility.

**Gate:** PRD §11 acceptance criteria all true.

## 5. Order of work

| Milestone | Contents | Exit |
|---|---|---|
| **M1 — Honest pilot** (now) | W8.3–4 (labels, goal tagging), W1.1 Kaggle bridge, pilot runs on the corpus | Facility can process + review + share; goals 100 % via tagging |
| **M2 — Ground truth** | W0 recording + labelling | Held-out futsal match labelled |
| **M3 — Vision core** | W2 detection, W3 tracking, W4 team | §2 detection / tracking / team gates |
| **M4 — Events** | W7 (SoccerTrack fine-tune → futsal fine-tune), W6.2 goal posts | §2 event gates on held-out match |
| **M5 — Identity** | W5 jersey numbers + roster | Identity gate |
| **M6 — Scale** | W1.2 production GPU, W9 multi-camera, W10 hardening | PRD §11 launch criteria |

M3 and M4.1 (SoccerTrack fine-tune) can start before M2 finishes; their futsal gates cannot pass until it does.

## 6. Data and model licences (for anything we ship)

| Source | Licence | Use |
|---|---|---|
| SoccerTrack v2 | CC BY 4.0 | Train + ship (attribution) |
| Roboflow Futsal-Object-Detection_v2 | CC BY 4.0 | Train + ship |
| Kaggle DFL ball / clip re-uploads | CC0 / unknown (DFL footage) | Evaluation; check before shipping weights trained on it |
| SoccerNet (GSR, jersey, BAS) | research only | Evaluation only |
| T-DEED code | GPL-3.0 | OK server-side (cloud); distributing the backend (on-prem) would bind it to GPL |
| T-DEED released checkpoints | trained on SoccerNet (research only) | Measurement only — the shipped event model is retrained from an ImageNet backbone on SoccerTrack (+ our labels) |
| jersey-number-pipeline | CC BY-NC 3.0 | Not usable commercially |

## 7. Needed from the product owner

- Pilot venue access for recording (W0), and goal times written down per match.
- A few hours of staff time to label 2 matches in the review screen.
- Budget decision for the production GPU provider (W1.2).
