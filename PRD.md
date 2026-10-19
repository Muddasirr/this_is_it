# Facility-first futsal analytics — finalized PRD and implementation plan

## 1. Executive decision

Build a cloud-based, post-match futsal analytics platform for facilities using two existing cameras with partial overlap. Facilities own the footage, register players through a roster, and upload completed matches for automated analysis. Staff review uncertain events before publishing player reports, statistics, and highlights.

The first release should optimize for reliable match analysis, not the number of AI features. Two-camera synchronization, player identity continuity, and trustworthy event attribution are the main technical risks.

## Locked product decisions

Initial customer and sport

Futsal facilities; 10–50 matches per facility per month.

Video capture

Two existing cameras, partially overlapping views. Facility-owned recordings.

Processing

Cloud-based, asynchronous, post-match processing. Live analytics deferred.

Player identification

Staff enter a roster and shirt numbers; AI proposes identities, and staff correct mismatches.

Quality policy

Auto-accept only validated, high-confidence events; send uncertain cases to review.

Not yet decided: pricing, monthly cloud budget, exact camera models, footage format/quality, and achievable model accuracy. These should be settled through a short technical validation and paid-pilot discussions rather than assumptions.

## 2. Product requirements

### Primary user journey

### 1

Set up facility and match

Create teams, register players and shirt numbers, enter match details.

### 2

Upload two recordings

Attach camera A and B videos; validate duration, frame rate, quality and sync.

### 3

Process in the cloud

Detect players and the ball, track movement, infer events, and associate identities.

### 4

Review uncertain events

Confirm goals, shots, passes and player attribution; fix roster mappings.

### 5

Publish results

Generate match dashboard, player reports, and shareable highlight clips.

### MVP scope

| Priority         | Included                                                                                                                                                                                                                                                                                                                                                    |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 — required    | Facility accounts, match and roster setup, two-video upload, video validation, synchronization and manual offset correction, asynchronous processing, player tracks, roster-based identity correction, review timeline, goals, shots, basic passes, match dashboard, player reports, event clips, share permissions, job status, deletion and cost tracking |
| P0 — reliability | Confidence scores, event evidence clips, review audit trail, retry-safe processing, duplicate-event prevention, clear “estimated / confirmed / unavailable” labels                                                                                                                                                                                          |
| P1 — after pilot | More reliable assists, possession estimates, defensive actions, goalkeeper saves and distribution, calibrated movement maps, historical player comparisons, exports and facility integrations                                                                                                                                                               |
| Deferred         | Live analysis, automatic tactical coaching, 3D reconstruction, guaranteed cross-camera re-identification, 11-a-side football, mobile apps and advanced automated highlight editing                                                                                                                                                                          |

Important scope constraint: “Passes” in P0 means a limited, evaluated pass-event capability. Do not promise complete pass networks or precise possession statistics until footage tests establish that the models can deliver them consistently.

## 3. Final architecture decisions

### Recommended architecture

Use a modular monolith for the product API and separate, independently scalable video-processing workers. This is simpler to maintain than microservices for a small team, while allowing GPU-heavy jobs to scale separately.

Web application

React + TypeScript · facility dashboard · match setup · review UI

Application API

Authentication · facilities · rosters · matches · signed upload URLs · review actions

PostgreSQL

Metadata, events, identities, audit log

Private object storage

Original videos, proxies, clips, artifacts

Job queue + workflow orchestrator

Retries · checkpoints · progress · cancellation · idempotency

CPU workers

FFmpeg · validation · synchronization · clip rendering

GPU workers

Detection · tracking · event inference

### Technology choices

| Area             | Decision                                                                | Reason                                                                |
| ---------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Web              | React + TypeScript                                                      | Fast development of dashboard and review tools                        |
| API              | Python + FastAPI                                                        | Convenient integration with computer-vision libraries                 |
| Database         | PostgreSQL                                                              | Relational data, transactions and auditable corrections               |
| Video storage    | Private S3-compatible object storage                                    | Large files should not live in the relational database                |
| Media processing | FFmpeg                                                                  | Probe, normalize, synchronize and cut videos                          |
| Vision           | OpenCV plus evaluated pretrained detection/tracking models              | Start with existing models; fine-tune only when footage justifies it  |
| Jobs             | Managed queue or Redis-backed queue with durable job state              | Keep uploads and API requests independent from long-running inference |
| Deployment       | Managed database/storage and on-demand GPU compute                      | Avoid operating a permanently idle GPU server                         |
| Monitoring       | Structured logs, job metrics, error tracking and per-match cost records | Diagnose failures and measure unit economics                          |

Specific vendors are intentionally left open until the footage and cost tests. Use a managed PostgreSQL service, private object storage, and a GPU provider that supports short-lived workers. Keep model inference behind a stable internal interface so models and vendors can be changed without redesigning the product.

### Processing workflow

1. Ingest: verify file type, duration, resolution, frame rate, audio stream, orientation and corruption.
2. Normalize: generate a low-resolution proxy for review; retain original footage under the facility’s retention policy.
3. Synchronize: estimate a relative time offset between cameras and store the offset, confidence and correction history.
4. Detect and track: process each camera independently first; track players and ball within each view.
5. Resolve identity: map tracks to roster entries using shirt numbers, team/color cues, temporal evidence and staff corrections.
6. Infer events: identify candidate goals, shots and passes, attaching timestamps, involved tracks, source camera and confidence.
7. Fuse observations: link the same event observed by both cameras so it is counted once.
8. Review: auto-accept only event classes whose precision has been validated; put uncertain events into a review queue.
9. Publish: derive statistics from canonical events, then render clips, player reports and match summaries.

### Data and processing rules

Use the following core entities:

- `facilities`, `facility_users`
- `teams`, `players`, `match_rosters`
- `matches`, `camera_sources`, `video_assets`, `camera_syncs`
- `processing_jobs`, `processing_runs`, `model_versions`
- `player_tracks`, `track_identity_links`
- `match_events`, `event_observations`, `event_participants`
- `event_reviews`, `event_corrections`, `highlight_clips`
- `usage_records`, `retention_policies`

Non-negotiable rules:

- Keep timestamps in a canonical match timeline; preserve each camera’s original timestamps and applied offset.
- A track ID is scoped to its camera and processing run. It is not automatically a globally unique player identity.
- One real-world event must have one canonical event record, even if both cameras observe it.
- Every published statistic must be traceable to its source event and any staff correction.
- Model or configuration changes must not silently overwrite staff-reviewed results. Reprocessing should create a new run and reconcile reviewed events.
- Store confidence and provenance separately from a simple “confirmed” flag.
- Use signed, expiring URLs for uploads and video playback; enforce facility-level access on every API request.

## 4. Camera installation and recording requirements

The facility already has two cameras, so the first step is a camera suitability audit, not an immediate hardware purchase. Existing CCTV may be designed for security coverage rather than sports analysis.

### Placement and coverage

[Loreto FESD Arcadi (@cd_arcadi) on X](https://images.openai.com/static-rsc-4/Pl9WyjdXeyijgTu7kp8569-KTNcFbPjMW1dD_3ae6mUpWrmOSPujBFDcISivV1D9mUhRVHmkCpFFCKL0tPP5jxmIDdPJja0fWwqFm9ZaE3p6HKllZRGq2zByez0KzPHib_VFGA-IRQIu7Odv4E_fqxlFJbCiM8Ftk5hJZaBtlfI?purpose=inline)

[Checking your browser - reCAPTCHA](https://images.openai.com/static-rsc-4/d7F2-agi7k_4ej3Ew0Il0h6lXDJ3MQXsPRFjtbASrIS3ZvrGhep--n_sTBTOezEHlYUdvKrE_qD54C-e4TmlZE9UJc7dwkWxLmE556EKzYDiispVRdX_BcLhKydr_7nqvI3OA3eLwA91qXQNEOOTEW-kqSYrN3zSSmtNA5WGUf8?purpose=inline)

[O nosso 𝗖𝗔𝗡𝗔𝗟 com as nossas 𝗖𝗢𝐑𝐄𝐒 💛❤️💙 Atlético Clube de Portugal - Site Oficial](https://images.openai.com/static-rsc-4/DSaGi2V7ANFSbFs7bh90GWOmaJH38XARdD6WxeTeY-alLKoYBK-g3Z67yMEgkoMBgMUcKua_hiDf-3lfM89QfbnMjlYHZt_T9bK3_jOZzRvXdGrT1XcHUie6OEnCqNNKHEMa1KbUCmxYWw0iASS_v3KhxClaSJ5I9tA13fJwYQU?purpose=inline)

8

The images are examples of the intended coverage concept, not a prescribed installation layout.

- Camera A: primary wide view of as much of the court as possible.
- Camera B: complementary angle that covers important areas obscured in Camera A, particularly goal areas and zones where players overlap.
- Mount cameras securely, high enough to reduce player-to-player occlusion, and outside ball/player impact zones.
- Prefer a stable, wide view over frequent PTZ movement. If a camera pans, tilts, or zooms, its changing field of view adds tracking complexity.
- Minimize glare, flicker, motion blur and extreme backlighting. Check image quality under actual match lighting.
- Keep both goal areas and boundary lines visible where feasible.
- Partial overlap is acceptable for the first version, but the cameras must share enough visible action to support synchronization and event cross-checking.

Do not require full 3D court reconstruction for the MVP. Begin with independent 2D camera views and a shared timeline. Partial overlap means some players and events will only be visible in one camera; the product must be able to report reduced confidence rather than invent missing evidence.

### Recommended recording profile

These are pilot targets, not claims that every existing CCTV system can meet them.

| Requirement   | Target for pilot                                               | Minimum validation                                                  |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| Resolution    | 1920×1080 per camera                                           | Test whether players and ball remain visible at the far end         |
| Frame rate    | 25–30 FPS preferred                                            | Test 15 FPS only as a fallback; fast ball motion may be missed      |
| Codec         | H.264 preferred                                                | H.265 is acceptable if the processing pipeline supports it reliably |
| Recording     | Continuous match recording                                     | Include warm-up/pre-match and post-match buffer where possible      |
| Camera motion | Fixed view                                                     | Record any PTZ movement or zoom changes                             |
| Exposure      | Fast enough to limit motion blur                               | Test at peak match speed and actual lighting                        |
| Time metadata | Preserve timestamps and time zone                              | Confirm both cameras’ clock accuracy and drift                      |
| File handling | Original recording retained until processing passes validation | Check file completeness and playback before deleting anything       |

If a camera only produces low-resolution or low-frame-rate footage, don't assume AI enhancement can recover information that was never captured.

### Synchronization requirements

With existing cameras, the most likely setup is independent recorders and clocks. Therefore:

1. Capture a common visible action at the start, such as a distinctive ball movement, whistle, or synchronized clap.
2. Estimate the relative time offset using timestamps and visible action.
3. Allow staff to adjust the offset in the review UI.
4. Validate the offset again later in the recording to detect clock drift.
5. Store the offset, method, confidence, and any manual correction.

Do not rely solely on file creation times: camera clocks may be wrong, time zones may differ, and recording systems may start files at different times.

A provisional pilot target is to achieve synchronization within approximately 100–200 ms when possible. The acceptable tolerance should be validated against event type and the frame rates of the actual recordings; a goal event may tolerate a different alignment error than ball-contact attribution.

### Installation acceptance checklist

- Both cameras can record a complete match without interruption.
- Files can be exported without proprietary playback software or unsupported encryption.
- The court and goal areas are sufficiently visible in the actual recordings.
- The ball is discernible in representative sections of play.
- The cameras' time offset can be estimated and manually corrected.
- The pipeline can decode the selected codecs.
- Recording starts/stops and missing segments are detected.
- The facility approves processing and retention of the footage.
- A full test match has passed ingestion, synchronization and playback checks.

## 5. Workload and cloud cost model

The unknown cloud budget should remain a measured variable. Do not select a fixed monthly infrastructure commitment until the first end-to-end benchmark is complete.

### A. Expected monthly workload

Planning assumptions: one match produces 60 minutes of footage per camera, both cameras record continuously, and average recording bitrate is 4–8 Mbps per camera. These assumptions must be replaced with measurements from the facility.

### Monthly workload calculator

Matches per month

### 25

10

30

50

Recording minutes per camera

### 60

40 min

60 min

90 min

Average bitrate per camera

### 6 Mbps

2 Mbps

6

10 Mbps

Retention period

### 30 days

7 days

30 days

90 days

New raw footage/month

# 135 GB

Approx. raw storage at steady state\*

# 135 GB

\*Assumes evenly distributed uploads, no extra proxies or clips, and a retention window expressed in 30-day months. Decimal GB is used; actual container overhead and bitrate variation will change the totals.

For the midpoint scenario of 25 matches, 60 minutes, and 6 Mbps per camera, raw uploads are approximately 135 GB/month. At 50 matches, 90 minutes, and 10 Mbps, uploads rise to about 675 GB/month. Proxies, clips, retries and longer retention add to storage and transfer.

### B. Cost model

Use the following model for every match:

\\[ \begin{aligned} C\_{\text{match}} ={}& C\_{\text{inference}} + C\_{\text{media}}\\\ &+ C\_{\text{storage}} + C\_{\text{delivery}}\\\ &+ C\_{\text{review}} + C\_{\text{overhead}} \end{aligned} \\]

Where:

- Inference: GPU worker time, model loading, and CPU preprocessing.
- Media: upload processing, transcoding, synchronization and clip generation.
- Storage: original footage, proxies, clips, model artifacts and logs.
- Delivery: playback and downloads, especially repeated full-match viewing.
- Review: staff time spent confirming or correcting events.
- Overhead: database, queue, monitoring, backups, retries and idle capacity.

A practical compute estimate is:

\\[ C\_{\text{inference}} = H\_{\text{GPU per match}} \times R\_{\text{GPU/hour}} \\]

Measure GPU hours using the actual models, resolution, frame sampling, and two-camera concurrency. Benchmarking a single still image is not sufficient.

### C. Illustrative planning scenarios

The following are initial budget envelopes for planning, not vendor quotes or measured predictions. They assume on-demand compute, limited retention, a small managed application stack, and a pilot that is still being optimized.

| Monthly facility volume | Early-stage cloud envelope | Main risk                                        |
| ----------------------- | -------------------------- | ------------------------------------------------ |
| 10 matches              | $50–$200/month             | Minimum managed-service costs dominate           |
| 25 matches              | $75–$350/month             | Inference efficiency and media delivery          |
| 50 matches              | $125–$600/month            | GPU utilization, retries, retention and playback |

These envelopes exclude dedicated installation hardware, taxes, engineering salaries, and substantial human review labor. They could be exceeded if models require expensive GPU time, videos are repeatedly downloaded, or storage is retained indefinitely. They could also be lower after optimization and credits. Do not present these figures to a facility as a guaranteed price.

### D. Cost-control decisions

1. Use on-demand GPU workers, not a continuously running GPU instance.
2. Process the two camera feeds in parallel only if benchmarks show the throughput gain justifies the GPU cost.
3. Test 5, 10 and 15 FPS sampling on representative footage before selecting a default.
4. Cache intermediate results by video checksum, model version and configuration.
5. Retry only failed stages, not the entire match pipeline.
6. Generate low-resolution proxies for review; create full-quality clips only when needed.
7. Apply explicit retention defaults and a deletion workflow for original recordings.
8. Track processing minutes, GPU seconds, storage GB-days, egress, retries and staff review minutes per match.
9. Set a per-match spending threshold and stop or queue jobs that exceed it pending investigation.

### E. Business-model decision

Pricing remains undecided. Test per-match pricing first as a hypothesis, while comparing it against a facility subscription. With 10–50 matches per month, per-match pricing aligns payment with usage, while a subscription may make revenue more predictable for facilities with regular volume.

Calculate gross margin only after including variable cloud costs and human review:

\\[ \text{Gross margin} = \frac{\text{Revenue} - \text{Variable cost}}{\text{Revenue}}\times100 \\]

The facility pilot should reveal whether customers pay for the analytics, how much staff time the review process consumes, and whether the resulting margin can support the service.

## 6. Implementation plan and realistic workload

For a solo developer or small team, a narrow controlled beta is plausible in 12–16 weeks, but only if existing models are usable on the footage. Reliable cross-camera identity resolution, ball tracking, and event attribution may take substantially longer. This estimate assumes one full-time technical owner, some part-time product/QA help, and access to a facility for testing.

| Phase                  | Duration    | Deliverable and exit gate                                                                                      |
| ---------------------- | ----------- | -------------------------------------------------------------------------------------------------------------- |
| 0. Feasibility         | Weeks 1–2   | Test recordings from both cameras; prove decoding, sync, player/ball visibility and basic inference throughput |
| 1. Facility foundation | Weeks 3–5   | Accounts, roster, match creation, uploads, private storage, job tracking and retention                         |
| 2. Vision pipeline     | Weeks 6–8   | Player detections/tracks, camera-specific IDs, roster mapping and manual correction                            |
| 3. Events and review   | Weeks 9–11  | Candidate goals/shots/passes, event clips, review UI, canonical events and basic statistics                    |
| 4. Controlled pilot    | Weeks 12–14 | Run real matches end-to-end, track quality/cost, fix critical failure modes                                    |
| 5. Hardening           | Weeks 15–16 | Reliability, security, report polish, pilot feedback and go/no-go decision                                     |

The schedule is a planning baseline, not a promise. Do not start full product implementation before Phase 0 has passed. If ball visibility or player identity is inadequate, revise the capture setup or narrow the MVP.

### Team and effort estimate

A lean implementation is roughly 16–24 person-weeks of engineering effort, plus facility coordination, model evaluation and pilot support. This is a range because model reuse and footage quality dominate the uncertainty.

- Product/full-stack engineering: approximately 8–12 person-weeks.
- Computer vision and video pipeline: approximately 6–10 person-weeks.
- QA, security, deployment and operational tooling: approximately 2–4 person-weeks.
- Facility setup, annotation and match review: additional part-time work throughout.

One person can cover several roles, but the calendar will expand if the developer is also responsible for all data annotation, installation, support and product design.

## 7. Facility pilot plan

Run the first pilot with one cooperative futsal facility, two existing cameras, and real match recordings. The pilot is primarily a feasibility and willingness-to-pay experiment—not a broad commercial launch.

### Pilot stages

1. Week 1 — Camera and footage audit

   Collect 2–3 representative recordings, including fast play, crowded goal areas, occlusions, and both halves. Confirm export access, image quality, synchronization and recording completeness.
2. Weeks 2–3 — Ground-truth dataset

   Manually annotate a small set of full sequences and event clips. Record goals, shots, selected passes, visible player identities, and ambiguous cases. Split evaluation data by match—not by randomly mixing frames from the same match.
3. Weeks 4–6 — End-to-end prototype

   Process both camera recordings, generate tracks and event candidates, provide roster correction, and publish a basic report. These are overlapping engineering activities, not additional weeks on top of the implementation schedule.
4. Weeks 7–10 — Shadow operation

   Process real matches without relying on the AI output for official results. Staff compare AI predictions with manually verified outcomes and log failures, processing time and correction effort.
5. Weeks 11–14 — Limited user pilot

   Let facility staff use the review and reporting workflow. Test whether players value the output, whether staff can complete reviews quickly, and whether the facility will pay.

### Pilot success criteria

Use the first dataset to establish the measurement protocol, then agree on numeric thresholds before judging the model. Suggested initial gates:

| Dimension       | Measurement                                                  | Go/no-go question                                         |
| --------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Capture         | Usable footage percentage, missing segments, ball visibility | Can the existing cameras support the intended stats?      |
| Synchronization | Offset error at start, middle and end                        | Can the two views be aligned reliably?                    |
| Tracking        | Track continuity, identity switches, player coverage         | Can staff follow players through realistic occlusions?    |
| Events          | Precision and recall by event type                           | Are goals, shots and passes accurate enough to be useful? |
| Identity        | Correct roster attribution and unknown-player rate           | Can reports be trusted at player level?                   |
| Review          | Human correction minutes per match                           | Can the review workflow scale to 50 matches monthly?      |
| Operations      | Processing time, failed jobs, retries and cost per match     | Can a match be processed economically and reliably?       |
| Commercial      | Facility usage, repeat usage, willingness to pay             | Is this solving a sufficiently valuable problem?          |

Do not use a single overall accuracy score. A system that detects many events but assigns them to the wrong player may produce attractive yet misleading player reports. Report results separately for goals, shots, passes and identity attribution.

Pilot gate: continue to broader release only when the capture pipeline is stable, the core events meet agreed quality thresholds, staff review is manageable, and projected unit economics work at the facility's expected match volume.

## 8. Final MVP backlog

The backlog below is ordered by dependency. Build the earliest items first; do not start advanced statistics while ingestion, synchronization and identity remain unreliable.

## Implementation checklist

0 / 30 complete

### A. Facility and access

Facility workspace and staff authentication

Facility-scoped roles and permissions

Create matches, teams, roster and shirt-number mapping

Match status, ownership and deletion policy

### B. Video ingestion

Upload two recordings using signed URLs

Validate codec, duration, frame rate and resolution

Private object storage and low-resolution review proxies

Processing job state, progress, retry and cancellation

Camera sync offset, confidence and manual adjustment

Detect incomplete or incompatible recordings

### C. Vision and identity

Run player detection and per-camera tracking

Evaluate ball detection on real match footage

Map tracks to roster entries with manual correction

Preserve unknown-player and uncertain-identity states

Record model versions, confidence and source evidence

### D. Events and review

Generate candidate goal and shot events

Generate limited pass candidates where quality allows

Deduplicate events observed by both cameras

Review timeline with evidence clips and accept/reject/edit actions

Maintain review audit trail and preserve corrections

### E. Reports and sharing

Match dashboard from canonical events

Player reports with attribution confidence

Event-based clip generation and playback

Share reports/clips with authorized users

Clearly label estimates, confirmed events and unavailable metrics

### F. Operations and launch

Per-match compute, storage and review cost tracking

Job logs, error reporting and operational alerts

Retention enforcement and deletion verification

Evaluation dataset and per-event quality report

Facility pilot onboarding and feedback collection

Reset checklistCopy backlog

## 9. Unresolved technical blockers

These are the items that must be investigated before committing to full MVP delivery.

Blocker 1 — Suitability of existing camera footage

Critical

Security footage may not capture the ball, jersey numbers, or distant players clearly enough. Low resolution, blur, glare and occlusion can invalidate downstream analytics.

Resolution: test real footage before model selection or product build-out. If needed, adjust camera settings, reposition cameras, or narrow the supported stats.

Blocker 2 — Partial-overlap cross-camera identity

A player can disappear from one view and appear in the other. Similar uniforms and partial visibility make identity matching difficult. Shirt numbers may not be visible consistently.

Resolution: use camera-specific tracks, roster mapping and manual identity correction first. Do not promise seamless cross-camera re-identification in P0.

Blocker 3 — Camera synchronization and drift

Existing cameras may have unsynchronized clocks or export timestamps. A fixed initial offset may not remain accurate throughout the match.

Resolution: test offset estimation at multiple points in a full recording and support manual correction. If reliable sync is impossible, treat the camera feeds as independent evidence rather than assuming frame-level alignment.

Blocker 4 — Event attribution and evaluation data

A detected player or ball does not automatically establish who passed, shot, scored or assisted. These labels require event-level ground truth.

Resolution: manually annotate representative matches, evaluate each event type separately, and keep uncertain attributions reviewable. Do not publish fabricated precision.

Blocker 5 — Actual cloud cost and throughput

GPU hours per match, model loading overhead, frame-sampling requirements and retry rates are unknown.

Resolution: benchmark two full camera feeds on the intended pipeline. Record actual processing time and cost before committing to pricing or a monthly budget.

### Additional launch dependencies

- Footage permission: facility ownership does not automatically resolve every privacy or player-consent obligation. Confirm applicable requirements, permitted use, retention and deletion before launch.
- Existing-camera export: verify the facility can export usable recordings in a supported format without manual work that becomes a bottleneck.
- Operational review: estimate the human time required to correct identities and event labels. If each match needs extensive manual work, the economics may not support the proposed service.
- Statistics definitions: document exactly what counts as a pass, shot, goal and assist. Define how disputed or invisible events are handled.

## 10. Final recommendation

Proceed with Phase 0 only as the immediate commitment.

1. Obtain representative two-camera footage from the target facility.
2. Test decoding, synchronization, ball visibility, player tracking and inference speed.
3. Annotate a small evaluation dataset.
4. Measure quality and cost on full matches.
5. Only then lock the MVP model choices, numerical acceptance thresholds, hosting provider and commercial offer.

The best first product is not a complete futsal intelligence platform. It is a dependable workflow that turns two facility-owned recordings into reviewable goals, shots, selected passes, roster-linked player reports and useful clips. Advanced possession, defensive analytics, movement maps and live analysis should follow only after the core pipeline proves trustworthy and economically viable.