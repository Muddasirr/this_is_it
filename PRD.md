# CourtLens — Production PRD

**Product:** CourtLens  
**Domain:** Facility-first futsal analytics (post-match)  
**Status:** Production requirements (v1.0)  
**Related:** `ARCHITECTURE.md` (pipeline + stat definitions), `PILOT.md` (single-court pilot constraints)

---

## 1. Vision and product thesis

CourtLens turns facility-owned match video into **reviewable, publishable player and match analytics**.

Facilities upload completed match recordings (1–2 fixed cameras). The platform detects and tracks players and ball, proposes roster identity (kit colour + jersey number), infers possession-centric events, and lets staff confirm uncertain cases before publishing dashboards, player reports, and shareable highlight links.

**Production bar:** a facility can run weekly matches with minimal engineer involvement, trust labelled stats enough to share with coaches/players, and the operator can run the service with measured cost, uptime, and data controls.

This is **not** a live broadcast Opta replacement, 3D reconstruction product, or 11-a-side football platform in v1.

---

## 2. Locked product decisions

| Decision | Production choice |
| -------- | ----------------- |
| Sport / customer | Futsal facilities (indoor courts); multi-tenant SaaS |
| Capture | Facility-owned fixed cameras; **1 or 2** views supported (1-cam fully supported) |
| Timing | **Post-match, asynchronous** cloud processing; live deferred |
| Identity | Auto: kit colour clustering + jersey OCR when readable; staff override always wins |
| Events | Pass, shot, goal (and assist when attribution allows) from **possession changes**; see `ARCHITECTURE.md` |
| Quality labels | Every published metric: `estimated` \| `confirmed` \| `unavailable` |
| Review | High-confidence classes may auto-accept after validation gates; uncertain → review queue |
| Sharing | Facility-controlled share links for reports/clips; no public index by default |
| Compute | On-demand GPU workers for detection/tracking; CPU OK for media/clip jobs; no always-on idle GPU |
| Single source of truth | Stats derive only from **canonical events** + track metrics linked through identity |

**Still commercial variables (not engineering assumptions):** list pricing, SLA tier packaging, exact GPU vendor. Settle via measured unit economics after production benchmarks.

---

## 3. Users and jobs-to-be-done

| Persona | Primary jobs |
| ------- | ------------ |
| Facility admin | Create facility, invite staff, set retention, manage billing contact |
| Match operator / coach staff | Create match, upload video(s), start processing, review events/identity, publish |
| Coach / player (consumer) | Open share link, view match + personal stats and clips (no edit) |
| Platform operator (CourtLens) | Monitor jobs, costs, failures; manage model versions; support facilities |

### Primary journey (production)

1. **Onboard facility** — workspace, roles, teams, optional pre-registered roster.  
2. **Create match** — metadata, optional Camera B, retention policy.  
3. **Upload** — signed URLs; validate codec, duration, FPS, resolution, integrity.  
4. **Process** — ingest → (optional sync) → detect/track → identity → events → fuse → stats.  
5. **Review** — fix identity, accept/reject/edit events; audit trail.  
6. **Publish & share** — match centre + player reports + clips; share token.  
7. **Retain / delete** — policy enforcement; verified deletion.

---

## 4. Production scope

### 4.1 Must ship (P0) — production v1

**Multi-tenant product**

- Facility workspaces, staff auth (email/password or SSO-ready design), roles: `admin`, `operator`, `viewer`.
- Facility-scoped authorization on every API and media URL.
- Match lifecycle: draft → processing → needs_review → published → archived.
- Soft-delete + hard-delete with retention jobs.

**Ingestion & media**

- 1 or 2 camera uploads via signed, expiring URLs.
- Validation: container/codec, duration, FPS, resolution, corruption, incomplete files.
- Proxy generation for review; originals in private object storage.
- Job status: queued, running, stage progress %, retry, cancel; idempotent stages.
- Optional Camera A/B sync offset with manual correction (2-cam).

**Vision & identity**

- Player + ball detect/track (stable `VisionBackend` interface).
- Dense track series, kit colour features, jersey OCR when available.
- Proposed roster (Kit A/B + shirt number); staff rename / reassign.
- Unknown / low-confidence identity states preserved — never silently invent certainty.

**Events, review, stats**

- Candidate pass / shot / goal / assist / dribble; cross-camera dedupe when 2-cam.
- Review UI: timeline, evidence clips, accept / reject / edit; full audit log.
- Match dashboard + per-player reports: goals, shots, passes, assists, dribbles, receptions, possession %, touches, minutes, relative speed, conversion %, movement + on-ball heatmaps.
- Clear `estimated` / `confirmed` / `unavailable` labelling in UI and exports.
- Publish + shareable report link (`/report/<token>`).

**Operations**

- Structured logs, error tracking, per-match cost records (GPU seconds, storage GB-days, egress, retries, review minutes).
- Model / config version stamped on every processing run.
- Reprocess creates a **new run**; staff-confirmed events are not silently overwritten.
- Health checks, queue depth alerts, failed-job alerts.
- Secrets via env / secret manager; no credentials in repo.

### 4.2 Should ship soon after v1 (P1)

- Calibrated pitch metres (homography) for true distance/speed.
- Stronger ball model or domain-tuned detector for wide CCTV.
- Assists as first-class confirmed metric; defensive actions; GK saves.
- Exports (CSV/PDF), webhook on publish, facility branding on share pages.
- SSO (Google / Microsoft), audit export for facilities.
- Batch GPU workers (e.g. Kaggle/Colab only for R&D — not production path).

### 4.3 Explicitly out of scope (v1)

- Live / real-time analytics during play.
- Guaranteed cross-camera re-identification without staff help.
- 3D reconstruction, automated coaching advice, 11-a-side football.
- Native mobile apps; full Opta-complete event taxonomy.
- Free always-on public GPU as production compute.

---

## 5. Quality and trust requirements

### 5.1 Labelling contract

| Label | Meaning | May appear in published share? |
| ----- | ------- | ------------------------------ |
| `confirmed` | Staff accepted or auto-accepted under validated gate | Yes |
| `estimated` | Model/heuristic output, not staff-confirmed | Yes, clearly marked |
| `unavailable` | Insufficient evidence (e.g. no ball) | Yes, as gap — not zero-filled as truth |

**Rule:** Player reports must not present estimated counts as official without the label. Wrong identity is worse than missing stats.

### 5.2 Production quality gates (per event class)

Numeric thresholds are **locked after** an evaluation set from real facility footage. Until then, all events default to review or `estimated`.

| Class | Gate to allow auto-accept | Until gate met |
| ----- | ------------------------- | -------------- |
| Goal | Precision ≥ agreed threshold on held-out matches | Always reviewable |
| Shot | Precision ≥ agreed threshold | Review or estimated |
| Pass | Precision/recall band agreed; noisy → review | Estimated + review |
| Identity (kit+number) | Attribution accuracy ≥ agreed threshold | Staff correction required before trust claims |

Report precision/recall **separately** for goal, shot, pass, and identity. Never a single “accuracy %”.

### 5.3 Honesty constraints (product copy & sales)

- Do not promise Opta-level pass networks or calibrated metres until homography ships.
- Do not promise readable jersey OCR on all CCTV; kit clustering + rename is the fallback.
- Sparse ball ⇒ possession/touches/shots degrade; UI must surface ball coverage.

---

## 6. Non-functional requirements (production)

### 6.1 Reliability & performance

| Metric | Target (v1) |
| ------ | ----------- |
| API availability | 99.5% monthly (excluding planned maintenance) |
| Upload success | Resumable or retry-safe; clear failure reasons |
| Processing SLO | p50 ≤ 1.5× realtime on GPU worker for 1×60 min 1080p @ configured stride; document CPU fallback separately |
| Job durability | Survives worker restart; stage checkpoints; at-most-once side effects via idempotency keys |
| Review UX | Evidence clip available for each candidate event |

### 6.2 Security & privacy

- TLS everywhere; signed expiring URLs for upload/playback.
- Facility isolation: no cross-tenant data in queries or URLs.
- Password hashing (or IdP); session/JWT expiry; role checks on mutations.
- Footage ownership stays with facility; CourtLens is processor under agreed terms.
- Retention defaults (e.g. 30 days) + facility-configurable; deletion verified.
- Player/personal data minimized; share links revocable.
- Audit trail for identity and event corrections (who/when/what).

### 6.3 Scalability

- Design for **10–50 matches / facility / month**, many facilities.
- Horizontal scale: API replicas, worker pool, object storage.
- Queue back-pressure; per-facility concurrency limits to protect cost.
- Cache artifacts by `(video_checksum, model_version, config_hash)`.

### 6.4 Observability

- Structured logs with `facility_id`, `match_id`, `job_id`, `run_id`, `stage`.
- Metrics: queue latency, stage duration, GPU util, failure rate, cost/match.
- Alerting: job stuck, error spike, spend threshold exceeded.

### 6.5 Cost control

Per-match cost model:

\[
C_{\text{match}} = C_{\text{inference}} + C_{\text{media}} + C_{\text{storage}} + C_{\text{delivery}} + C_{\text{review}} + C_{\text{overhead}}
\]

Controls:

1. On-demand GPU only.  
2. Configurable frame stride / imgsz with documented quality trade-offs.  
3. Retry failed **stages**, not whole pipeline by default.  
4. Proxies for review; full-quality clips on demand.  
5. Retention + deletion enforcement.  
6. Per-match spend ceiling → pause + alert.

---

## 7. Architecture (production)

### 7.1 Shape

**Modular monolith API** + **independently scalable workers** (media CPU workers, vision GPU workers). Avoid microservice sprawl until team/size demands it.

| Layer | Production choice |
| ----- | ----------------- |
| Web | React + TypeScript |
| API | Python + FastAPI |
| DB | Managed PostgreSQL |
| Queue | Redis or managed queue; durable job state in Postgres |
| Storage | Private S3-compatible object storage |
| Media | FFmpeg |
| Vision | Ultralytics YOLO + ByteTrack (swappable via `VisionBackend`) |
| Events / identity | Possession heuristics + kit/jersey clustering (`ARCHITECTURE.md`) |
| Deploy | Containerized API/workers; managed DB/storage; on-demand GPU |

### 7.2 Processing stages (canonical)

1. **Ingest** — validate media.  
2. **Normalize** — proxy + metadata.  
3. **Synchronize** — A/B offset (if 2-cam).  
4. **Detect & track** — persons + ball; colour hist; OCR; thumbs.  
5. **Identity** — cluster → Kit A/B + shirt → roster links.  
6. **Events** — possession-centric candidates.  
7. **Fuse** — dedupe across cameras.  
8. **Stats** — canonical events + track metrics.  
9. **Review / publish** — staff gate → share artifacts.

### 7.3 Core entities

`facilities`, `facility_users`, `teams`, `players`, `match_rosters`,  
`matches`, `camera_sources`, `video_assets`, `camera_syncs`,  
`processing_jobs`, `processing_runs`, `model_versions`,  
`player_tracks`, `track_identity_links`,  
`match_events`, `event_observations`, `event_participants`,  
`event_reviews`, `event_corrections`, `highlight_clips`,  
`usage_records`, `retention_policies`, `share_tokens`.

### 7.4 Non-negotiable data rules

1. Canonical match timeline; preserve per-camera timestamps + applied offset.  
2. Track IDs are scoped to camera + processing run — not global player IDs.  
3. One real-world event → one canonical event (even if seen on both cameras).  
4. Every published statistic traces to source event(s) and corrections.  
5. Reprocess = new `processing_run`; do not erase confirmed staff work.  
6. Confidence and provenance stored separately from confirmation state.  
7. Facility ACL on every read/write of media and match data.

---

## 8. Capture requirements (production guidance)

Facilities use existing CCTV; CourtLens does not sell cameras in v1. Publish a **camera readiness checklist** in onboarding.

| Requirement | Production target | Minimum |
| ----------- | ----------------- | ------- |
| Resolution | 1920×1080 | Players/ball visible at far end |
| Frame rate | 25–30 FPS | 15 FPS fallback (expect missed ball) |
| Codec | H.264 | H.265 if pipeline-validated |
| View | Fixed wide; both goals preferred | Stable FOV (no PTZ during match) |
| Cameras | 1 supported; 2 preferred for goals/occlusion | 1-cam OK with lower confidence |

Sync (2-cam): common visible action at start; estimate offset; staff adjust; re-check mid/late match for drift. Do not trust file mtimes alone.

---

## 9. Workload assumptions

Planning defaults (replace with facility measurements):

- 10–50 matches / facility / month  
- ~60 minutes / camera / match  
- ~4–8 Mbps average bitrate  
- Retention 7–90 days (default 30)

Illustrative raw volume (25 matches × 60 min × 6 Mbps × 2 cams) ≈ **135 GB/month** uploads before proxies/clips/retries. Production capacity planning must include proxies, clips, and multi-tenant peaks.

---

## 10. Implementation roadmap (production)

**Lock:** do not start the next phase until the prior **exit gate** passes. See `EVAL.md`.

| Phase | Outcome | Exit gate |
| ----- | ------- | --------- |
| **0 — Eval + data** | `scripts/eval/*`, GT template, Roboflow download script, corpus list | `run_detection_smoke.py` exits 0 |
| **1 — Detect/track** | YOLO+ByteTrack+ball; court format 5v5/6v6/7v7; thumbs | Facility smoke: ball + person tracks in `summary.json` |
| **2 — Identity** | Kit A/B, throttled OCR, **avatar per player** | Every proposed player has thumb; staff rename works |
| **3 — Possession** | Vidal-Codina PZ + gain/loss (Δv/Δθ when possible) | Possession segments when ball samples ≥ 80 |
| **4 — Events + stats** | Pass/shot/goal + touches/poss%/**per-player heatmap** | No shot flood; `score_events.py` runs; heatmaps on player rows |
| **5 — Product** | Upload → process → review → share (avatars + **player heatmaps**) | One docker/local match completes with share report |
| **6 (later)** | Homography metres, cam B/C fuse quality, SSO/exports | Separate PRD |

**Court formats:** `COURT_FORMAT=5v5|6v6|7v7` (field players per side). **Cameras:** 1 now; labels A/B/C ready. **Heatmaps:** image-plane grids per player + match aggregate; always `estimated` until calibrated pitch.

---

## 11. Acceptance criteria (production v1 launch)

Launch is allowed only when **all** are true:

1. **Facility self-serve:** upload → process → review → publish → share without engineering.  
2. **Isolation:** penetration-style checks show no cross-facility data leaks.  
3. **Jobs:** failures retry safely; stuck jobs alert; reprocess preserves confirmed events.  
4. **Trust UI:** estimated/confirmed/unavailable always visible on share pages.  
5. **Identity:** staff can fix kit/number/name; reports update accordingly.  
6. **Economics:** measured \(C_{\text{match}}\) at target volume leaves room for chosen pricing hypothesis.  
7. **Legal/privacy:** retention, consent language, and deletion path agreed with facility.  
8. **Eval:** held-out match set exists; per-class metrics published internally before any auto-accept.

---

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Ball invisible on wide CCTV | Broken shot/goal/possession | Ball-coverage meter; tune detector; camera guidance |
| Jersey OCR fails | Weak shirt numbers | Kit clustering + staff rename; never block publish |
| CPU-only processing | Slow / costly UX | On-demand GPU; documented stride trade-offs; queue SLAs |
| Over-trust of heuristics | Bad player reputation | Labels + review gate + audit |
| Cost spikes (retries, egress) | Margin failure | Spend ceilings, proxies, retention |
| Sync drift (2-cam) | Double-count / wrong fuse | Manual offset + mid-match validation |

---

## 13. Success metrics (product)

| Metric | Why |
| ------ | --- |
| Matches processed / facility / month | Adoption |
| Median staff review minutes / match | Ops scalability |
| % events confirmed vs rejected | Model + UI quality |
| Share-link opens / match | Downstream value |
| Job success rate (no engineer) | Production readiness |
| Gross margin after variable cloud + review | Business viability |

---

## 14. Recommendation

**Ship production v1 as a hardened, multi-tenant version of the current CourtLens pipeline:** reliable upload → GPU-accelerated track/identity/events → staff review → labelled publish/share — with ops, cost, and privacy controls.

Do **not** expand into live analytics, full Opta taxonomy, or calibrated metres until §11 acceptance criteria and §5.2 quality gates are met for the core loop.

Advanced features (homography, defensive stats, SSO, exports) follow in P1 once unit economics and trust labels hold on real facility volume.
