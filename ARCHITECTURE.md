# CourtLens architecture — tracking, identity, events, player stats

This document records how top football leagues produce match data, how CourtLens adapts that approach for facility futsal CCTV, and the concrete tech choices in this repo.

## 1. How the top-five leagues do it

| League | Tracking | Events / stats | Player identity |
| ------ | -------- | -------------- | --------------- |
| Premier League | Second Spectrum optical (~25 Hz XY for every player + ball) | Stats Perform / Opta event feed fused with tracking (Insight Feed) | Known lineups + multi-camera ID |
| La Liga | Optical (historically TRACAB / league providers) | Opta / Stats Perform-style eventing | Lineups + jersey + multi-view |
| Serie A | Optical / broadcast-augmented providers | Opta / Stats Perform eventing | Same |
| Bundesliga | 16–20 stadium cameras, positions every ~40 ms (~3.6M points/match) | ~1,600 events/match via **human operators** (speaker → writer → video QA) | Multi-camera + operators |
| Ligue 1 | Genius Sports optical + AI event detection | AI + ops tooling for clubs / broadcast | Venue camera arrays |

Common pattern across all of them:

1. **Dense positional tracking** of every player and the ball (not “detect once and guess”).
2. **Event semantics** built on **possession**: who controls the ball, when control changes, ball velocity/direction at contact.
3. **Identity from known squads** (lineups), jersey numbers, team colors, and multi-camera continuity — not from a single anonymous blob ID.
4. **Human QA** for official stats (Bundesliga especially). AI proposes; humans gate publishable truth.
5. **Player reports are derived from canonical events**, never from raw detections alone.

Published research (e.g. *Automatic event detection in football using tracking data*, arXiv:2202.00804) formalizes the same idea for automation:

- **Possession zone (PZ)** around each player — ball inside PZ ⇒ candidate controller.
- **Possession gain/loss** when the ball leaves A’s PZ and later enters B’s PZ, often with a ball speed/direction change at contact.
- Classify that transition as **pass / shot / clearance / duel** using geometry (toward goal?), team affiliation, and kinematics.

SkillCorner / StatsBomb-style products also treat **player possession sequences** as the spine that events hang on.

## 2. What CourtLens can and cannot copy

Facility futsal with 1–2 fixed CCTV cameras is **not** a 16–20 camera optical install.

| League capability | CourtLens P0/P1 approach |
| ----------------- | ------------------------ |
| Calibrated pitch XY meters | Image-plane XY from YOLO boxes (no full court calibration yet) |
| Multi-view re-ID | Per-camera tracks + appearance clustering; staff correction |
| Jersey OCR | Deferred; appearance + roster mapping first |
| Human event ops room | Facility review UI (accept / reject / edit) |
| Official Opta taxonomy | Subset: goal, shot, pass (+ assist later) |

We still follow the **same logical pipeline** leagues use: track → possess → events → identity → stats → review → publish.

## 3. CourtLens processing architecture

```
Upload A/B video
    → Ingest / normalize (FFmpeg)
    → Synchronize (offset estimate + manual fix)
    → Detect & track (YOLOv8n + ByteTrack, persons + sports_ball)
         · dense bbox series per track
         · jersey-region color histogram features
    → Resolve identity
         · cluster tracks by appearance + non-overlapping time
         · auto-create "Detected Player N" roster if none provided
         · map clusters → roster / staff corrects
    → Infer events (possession-centric heuristics)
         · pass = sustained possession A → flight → possession B
         · shot = release with high ball speed toward goal band
         · goal = ball enters extreme goal band after shot-like motion
    → Fuse (dedupe by type + time window across cameras)
    → Review gate (clips + estimated/confirmed labels)
    → Publish (match totals + per-player stats from canonical events)
```

### Tech stack (current)

| Layer | Choice |
| ----- | ------ |
| API | FastAPI + PostgreSQL + Redis jobs |
| Workers | Python worker process (`app.workers.runner`) |
| Detection / tracking | Ultralytics YOLOv8n + ByteTrack (CPU-capable) |
| Events | Possession heuristics in `app/services/events.py` (no trained event net yet) |
| Identity | Appearance clustering in `app/services/identity.py` |
| UI | React review + match dashboard |

Stable interface: `VisionBackend.detect_and_track` / `infer_events` so models can be swapped without rewriting product APIs.

## 4. Statistic definitions (CourtLens)

These definitions are intentional and must stay aligned with `events.py` and the review UI.

### Pass (estimated)

A **pass** is recorded when:

1. Player A has **sustained possession** (ball nearest to A within a possession radius for ≥ N consecutive samples).
2. The ball then leaves A’s possession zone (flight / carry gap).
3. Player **B ≠ A** gains sustained possession within a time window.
4. The transition is **not** classified as a shot (see below).

Attributed to the **passer** (role `passer`). Receiver is stored as role `receiver` for later assist/reception metrics.

### Shot (estimated)

A **shot** is recorded when a player releases the ball and the ball travels with **high image-plane speed** toward a **goal band** (top or bottom ~12% of the frame — orientation-agnostic for uncalibrated CCTV).

Attributed to the shooter (role `shooter`).

### Goal (estimated)

A **goal** candidate is recorded when, after shot-like motion, the ball enters/lingers in an extreme goal band. Always review-gated; precision is not claimed.

### Player-wise stats

```
by_player[player_id].pass  += events where participant.role in {passer, primary} and type=pass
by_player[player_id].shot  += events where participant.role in {shooter, primary} and type=shot
by_player[player_id].goal  += events where participant.role in {scorer, primary} and type=goal
```

Rejected events never count. Confirmed events override estimated labels in the UI.

## 5. Auto player detection (identity)

Without jersey OCR or multi-camera calibration, CourtLens uses a league-inspired **lightweight identity stack**:

1. **Track** persons with ByteTrack (short-term continuity).
2. **Appearance**: HSV histogram of the upper-body crop (jersey region) averaged over the track.
3. **Cluster / stitch**: merge tracks that do not overlap in time and have similar appearance → one logical player.
4. **Roster**:
   - If staff registered a roster → propose cluster → shirt mapping (best effort).
   - If no roster → **auto-create** `Detected Player 1…N` facility players + match roster rows so player-wise stats always appear.
5. Staff can rename / reassign in the Tracks & identity UI; confirmed links win over model proposals.

This mirrors pro workflows (AI proposes → humans correct) at facility cost.

## 6. Known limits (honest)

- Single wide CCTV often **misses the ball**; on `test_video_2` YOLO saw ~50 ball samples vs thousands of person samples. Passes then rely partly on proximity handoffs; **shots/goals stay near-zero** until ball recall improves (better camera, denser model, or SAHI). Prefer labeling thin shot stats **estimated** rather than inventing them from player motion (the old bug that produced ~40 false shots).
- Image-plane distances ≠ meters; thresholds are relative to frame size.
- Auto “Detected Player N” is appearance clustering, not jersey OCR — staff should rename/merge before publish.
- Team affiliation (teammate vs opponent) is weak without kit clustering — interceptions may be counted as passes until team cues improve.
- Do not claim Opta-grade accuracy. Publish only after review when stakes are high.

## 7. Evolution path

| Stage | Upgrade |
| ----- | ------- |
| Now | YOLO tracks + possession heuristics + appearance clusters |
| Next | Team kit clustering; court calibration; denser ball model / SAHI |
| Later | Trained event classifier on facility annotations; jersey OCR; cross-camera re-ID |
| Optional | Broadcast-style providers (SkillCorner-like) if facilities adopt better cameras |

## 8. Code map

| Concern | Module |
| ------- | ------ |
| Detect / track + appearance features | `backend/app/services/yolo_backend.py` |
| Possession → pass/shot/goal | `backend/app/services/events.py` |
| Cluster + auto roster | `backend/app/services/identity.py` |
| Job stages | `backend/app/workers/pipeline.py` |
| Stats API | `backend/app/api/review.py` (`GET …/stats`) |
| Product requirements | `PRD.md` |
