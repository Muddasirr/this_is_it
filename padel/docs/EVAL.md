# PadelLens — evaluation

Protocol: **PadelTracker100 cross-match.** Models used on the women's final (F) were trained on the men's final (M) and vice versa. Downstream classifiers (hits, strokes) are trained on one match's held-out detections and scored on the other. Numbers below are produced by `python -m padel_analyzer.eval.run_eval` and are copied here unchanged.

Ground truth definitions:
- **Ball:** PT100 box centre; a prediction is correct within 10 px. Frames labelled "no ball" count predictions as false positives.
- **Players:** PT100 boxes (IoU ≥ 0.5); floor positions = ankle midpoint through the PT100 homography (same definition as ours).
- **Hits:** centre of each PT100 shot window (consecutive frames with the same shot label); a hit is matched within ±7 frames.
- **Stroke type:** PT100 label of the matched window (Dropshot folded into Other).
- **Tracks / distance:** GT positions linked frame-to-frame by Hungarian assignment (2 per net side).

## 1. Court calibration (measured 2026-10-26)

Floor error = mean distance (m) between where our homography and the PT100 homography put the same image points, over a 0.5 m grid of the whole floor. Clicks simulated by projecting the landmarks with the PT100 H and adding Gaussian pixel noise (50 / 15 trials).

| Clicks | Click noise σ | Fit | + snap to lines | Gate |
| --- | --- | --- | --- | --- |
| 13 landmarks | 2 px | **0.030 m** (max 0.055) | – | ≤ 0.15 ✅ |
| 4 service-line ends | 2 px | **0.077 m** | – | ≤ 0.15 ✅ |
| 4 service-line ends | 6 px | 0.196 m | **0.075 m** | ≤ 0.25 ✅ |
| 13 landmarks | 6 px | 0.147 m | **0.071 m** | |
| 4 service-line ends | 10 px | 0.225 m | **0.117 m** | |
| 13 landmarks | 3 px | 0.039 m | 0.069 m | snap hurts precise clicks → optional in the UI |

Only the two service lines and the centre line are painted on a padel floor (the outline is glass), so line fitting alone cannot recover all 8 dof; the snap uses the clicks as a prior. Real product run (API, 7 clicks with 2 px noise): fit residual 0.053 m, floor error vs PT100 **0.070 m**.

## 2. Downstream logic on oracle perception (GT ball + GT pose as input)

Isolates tracking / event logic from detector quality; first 20,100 frames (the shot-labelled range) of each final.

| Feature | Metric | F | M | Notes |
| --- | --- | --- | --- | --- |
| 4-player tracker | floor position error, median / p90 | 0.029 / 0.088 m | 0.030 / 0.094 m | coverage 99.7 / 99.3 % |
| 4-player tracker | pair distance vs GT (same smoother) | −0.1 % / −0.2 % | −0.5 % / −1.4 % | 2,084 / 1,998 m GT (F) |
| Cut detection | segments found | 32 | 34 | PT100 = main-view rally clips concatenated |
| Ball linker | P / R @10 px with 20 % dropout + static look-alike + 10 % random FPs | 0.944 / 0.940 | | per-frame argmax: 0.27 / 0.30 |
| Hits (GBM, cross-match) | F1 @±7 frames | **0.876** | **0.927** | rule-based score: 0.68 / 0.67 |
| Stroke type (GBM, cross-match) | accuracy on GT hits / macro-F1 | 0.62 / 0.53 | 0.82 / 0.72 | forehand↔backhand confusions dominate F |
| Stroke type | accuracy on predicted hits (end-to-end) | 0.64 | 0.81 | |

### Identity across cuts (visual check, women's final, 16 of 32 segments)

`scripts/identity_sheet.py` → `data/eval/identity_sheet.jpg`: rows = segments, columns = linked persons. **Team linking correct in 16/16 segments** (black kit always A, red kit always B, including segments after end changes where the pair seen from behind looks maroon). Partners share a kit, so partner-level identity is not verifiable this way → pair-level stats are the reliable level, per-player identity is labelled estimated.

### Stroke type experiments (oracle perception)

| Model | F acc | M acc | Finding |
| --- | --- | --- | --- |
| GBM on pose window (4 offsets) + ball context | 0.62 | 0.82 | kept |
| + per-person handedness vote | 0.62 | 0.82 | slots ≠ persons across end changes, votes mixed |
| + per-hit racket hand (faster wrist) | 0.64 | 0.79 | wrist-speed hand estimate only ~65 % consistent |
| denser window (11 offsets) | 0.64 | 0.78 | no gain |
| temporal CNN over pose sequence + mirror augmentation | 0.43 | 0.78 | mirroring makes forehand ≡ other-hand backhand: pose alone cannot tell which hand holds the racket |

| **RGB clip classifier** (R(2+1)D-18, Kinetics-pretrained, 16 frames of the hitter crop, mirror aug; Kaggle `padel-stroke-rgb`) | **0.749** (macro-F1 0.66) | 0.741 (macro-F1 0.68) | sees the racket: F forehand↔backhand errors 39+37 → 21+17; "other" stays ~0 % (catch-all class) |

Conclusion: the remaining forehand/backhand error is a handedness problem, not a model-capacity problem → racket-hand evidence (racket detection or an RGB clip model on the hitter crop) is the next step (ROADMAP P1). Confirmed: the RGB clip model lifts the women's final from 0.62 to 0.75; pose GBM still wins on the men's final (0.82) → product fuses both (0.7 RGB + 0.3 pose/context); fusion weight to be measured on held-out probabilities.

## 3. Held-out perception (cross-match detectors)

Measured 2026-10-26 on Kaggle (`padel-gpu-pose` → `run_eval`), every frame of both finals (45,934 / 53,888), detectors trained on the *other* final. Hit / stroke rows use the shot-labelled range (first 20,100 frames).

| Feature | Metric | F (women's) | M (men's) | Gate / reference |
| --- | --- | --- | --- | --- |
| Players (YOLO11m-pose fine-tuned @1280) | box P / R, IoU 0.5 | **0.998 / 0.999** | **0.974 / 0.988** | ≥ 0.97 ✅ (HASHU7 99/99) |
| Players (stock yolo11m-pose) | box P / R | 0.818 / 0.535 | 0.937 / 0.664 | fine-tuning is required |
| Pose model val | box / keypoint mAP50 | 0.993 / 0.890 | 0.995 / 0.895 | |
| Floor position | median / p90 error | **0.051 / 0.140 m** | **0.053 / 0.153 m** | ≤ 0.30 m ✅ |
| Pair distance | vs GT (same smoother), per pair | −1.2 % / −0.1 % | −0.1 % / +1.7 % | ≤ 5 % ✅ |
| Ball (YOLO11s @1536), per-frame best ≥ 0.25 | P / R @10 px, all frames | 0.877 / 0.885 | 0.357 / 0.887 | M precision: PT100 leaves the ball unlabelled between points (only 36 % of M frames labelled) → in-rally metric below |
| Ball linker v1 (min_conf 0.08, fill 8) | P / R | 0.796 / 0.856 | 0.326 / 0.855 | too permissive: hurt precision |
| **Ball, in-rally frames** (PT100 rallies ±1 s; the ball is unlabelled between points) | P / R @10 px, per-frame best | 0.929 / 0.868 | 0.904 / 0.891 | |
| **Ball linker, tuned on the other final** (min_conf 0.35, fill ≤ 4 — same winner both ways) | in-rally P / R (F1) | **0.913 / 0.873 (0.893)** | **0.913 / 0.896 (0.904)** | P ≥ 0.85, R ≥ 0.80 ✅ (HASHU7 88/89, 90/84; lmrt 80 % recall) |
| Hits (GBM, cross-match), features on the recall-oriented ball track | F1 @±7 frames (P / R) | **0.880** (0.930 / 0.836) | **0.868** (0.893 / 0.844) | ≥ 0.85 ✅ (refs 0.88–0.91); rule score 0.59 / 0.57; on the precise display track: 0.855 / 0.878 |
| Stroke type, pose GBM | acc on GT hits / macro-F1 | 0.678 / 0.581 | 0.745 / 0.637 | |
| Stroke type, pose GBM | acc on predicted hits (end-to-end) | 0.727 | 0.796 | |
| Stroke type, RGB clip model | acc on GT hits | 0.749 | 0.741 | |
| **Stroke type, 0.7 × RGB clip + 0.3 × pose GBM** | acc on GT hits | **0.788** | **0.783** | ≥ 0.75 ✅ (HASHU7 0.81 / 0.82); fusion weight grid: best 0.9 (F) / 0.6 (M), 0.7 within 0.5 pt of both |
| Rallies (hits grouped, gap 4 s + cuts) | predicted vs GT-hit rallies | 33 vs 32 (+3 %) | 36 vs 39 (−8 %) | ±10 % ✅ — gap chosen looking at both finals (3.5 s: +22 % / +10 %; 5 s: 0 % / −18 %), not cross-tuned |
| Cut / segment detection | segments, whole match | 68 | 81 | same as with GT detections (70 / 81): PT100 is rally clips stitched together, so these are real cuts |

### Product run through the Kaggle backend

API (`PADEL_BACKEND=kaggle`) → `padel-jobs` dataset → `padel-infer` (T4) → report, on the 13.6 s PT100 sample clip with all production models: 7 of 8 GT hits found at the right time (the missed one is 0.4 s before the clip ends), all 7 stroke types match the labels, players tracked in 100 % of frames. The clip is from the women's final, which the production models trained on — this checks the path, not accuracy (accuracy = the held-out table above). All 11 unit tests pass on Kaggle (`padel-eval`).


## 4. Multimodal hit detection (audio + video), 11 unseen tournaments

Data: Decorte et al., *Multi-Modal Hit Detection and Positional Analysis in Padel Competitions* (CVPRW 2024) — 99 rallies from 11 WPT 2023 tournaments (indoor/outdoor; blue, grey, purple courts), 25 fps, 2,377 hit onsets annotated on the audio. Kaggle `padel-decorte-prep` → `padel-audio` (2026-10-27). Folds are grouped by **tournament** (4 folds), so every number is on venues the models never trained on — stricter than the paper's random rally splits.

| Hit detector | P | R | **F1 @±200 ms** | F1 @±280 ms | worst tournament (F1 @200 ms) |
| --- | --- | --- | --- | --- | --- |
| Video only (ball_v2 + pose_all features, GBM) | 0.840 | 0.826 | 0.833 | 0.857 | 0.751 Amsterdam |
| Audio only (CRNN probabilities → same GBM) | 0.942 | 0.915 | 0.928 | 0.937 | 0.868 Malmö |
| **Audio + video fused** | **0.948** | **0.929** | **0.938** | **0.947** | **0.888 Malmö** |
| Raw audio CRNN peaks (no classifier) | 0.833 | 0.801 | 0.817 @250 ms | — | — |
| *Paper (audio, random splits)* | | | *0.92 @250 ms* | | |

- Fusion beats each modality overall and lifts the hardest venues most (Vigo 0.77 → 0.91, Amsterdam 0.75 → 0.93, Valladolid 0.78 → 0.95).
- Raw CRNN peaks are poor at tight tolerances (0.14 @100 ms) because the labels span the whole hit sound (~0.4 s), so the probability peaks mid-sound; the GBM learns the offset from the audio features (`aud_p1`, `aud_p3`, `aud_dt`).
- Ball found in 73.8 % of frames on these 11 venues with `ball_v2`.
- **Modality dropout** (`padel-fusion`): training copies with the ball features blanked and with the audio blanked. Same accuracy with both signals (F1 0.939), 0.922 with no ball features, and **0.783 with no audio track** — plain fusion silently depended on audio (0.000 without it). Production hit model = fused + dropout.
- Club camera (YouTube KLICK court, 6 min, 60 fps, audio): video-only 21 hits / 16 rallies → fused **169 hits / 32 rallies** (avg 5.3 shots, longest 18), consistent with the on-screen score (≈ 4 games). No ground truth for this clip; per-player hit attribution looks unbalanced (partners 75 vs 21 shots) because near players' feet leave the frame → trust pair totals.
- Bug fixed on the way: for 60 fps video (sampled to 30) the audio timeline was stretched ×2 (frame step applied twice).
- One shared homography (PT100's) for all 11 broadcast venues: only affects court filtering / net side, the hit features are pixel-based.
- Production hit model (`hit_lgbm.txt`, 25 features): all 99 Decorte rallies + PadelTracker100 held-out rows (video-only, audio NaN) → uses audio when the video has a track, falls back to video otherwise.

### Who hit the ball (Decorte `hit_assignments`, 319 hits, Vigo; Kaggle `padel-hitter`, 2026-10-27)

Given the labelled hit frame, the hitter is the player whose wrist is closest to the ball (±3 frames); top/bottom = net side, left/right relative to the partner.

| | Ours | Paper |
| --- | --- | --- |
| Player | **0.897** (310 evaluated; 0.872 counting the 9 with no ball/player detection as wrong) | 0.837 |
| Team | **0.936** | 0.868 |

Main error: near-side hits credited to a far-side player (18 of 32 errors) — overheads, where a high ball near a near-side player projects next to the far pair in the image. Next: use the ball's direction after the contact.

## 5. Venue diversity (Roboflow Universe padel sets, CC BY 4.0)

Kaggle `padel-venues` (2026-10-26). Ball detector fine-tuned from `ball_all` on PT100 + 5 Roboflow ball sets (`ball_v2`), and a leave-one-venue-out copy without the largest set (`ball_lovo`). mAP50 on each set's test split:

| Model | padel-dataset (ll7pp) | home | padel-detection-5 | rbada | padel-fp1pf |
| --- | --- | --- | --- | --- | --- |
| ball_v2 (all sets) | 0.617 | 0.730 | 0.821 | 0.531 | 0.907 |
| ball_lovo (ll7pp unseen) | **0.248** | 0.804 | 0.826 | 0.707 | 0.914 |

Club camera (YouTube KLICK court, 720p60 wide-angle): ball found in 8 % of frames with `ball_all` → 50 % with `ball_v2`; hits 3 → 21 in 6 min (video only). Unseen-venue ball detection remains the weakest link → audio fusion (§4) carries hit detection there.

**Padel shot-type detector — not shipped.** On PadelTracker100 broadcast hits and on the club clip it produced **zero** player-shot detections (coverage 0 %; only "ball" boxes, many on lights): it only works on its own dataset's frames. Dataset numbers for reference — YOLO11s, padelcnn/padel-quret, 9.9k frames: test mAP50 0.942, mAP50-95 per class — bandeja 0.85, smash 0.79, forehand volley 0.77, backhand volley 0.75, forehand 0.75, off the wall 0.74, serve 0.74, backhand 0.72, lob 0.70, contrapared 0.70 (test split likely shares source videos with train → optimistic; not yet validated on PT100 / club footage).


## 6. Production run — full 25-minute match through the app (2026-10-27)

WPT Barcelona Master Final 2022, women's final (PadelTracker100 video, 45,934 frames, no audio track) → API → `padel-jobs` → `padel-infer` (T4) → report: completed end to end; players tracked 100 %, ball seen 78.7 %, 68 scene cuts handled. Sanity check against the labelled first 11 minutes (this match is in the production models' training data, so not a held-out number):

| Hit model on a video without audio | P | R | F1 | rallies (GT 30) |
| --- | --- | --- | --- | --- |
| audio+video model (dropout) running without audio | 0.78 | 0.96 | 0.86 | 30 |
| **video-only model (auto-selected when there is no audio)** | **0.93** | **0.91** | **0.92** | **30** |

→ the pipeline now picks the hit model per video: audio+video (F1 0.94 held-out) when the file has sound, the dedicated video model (F1 0.88 held-out) when it does not.

Production checks: Docker image builds and serves the app (health, token auth 401/200, static web), 16/16 unit + API tests pass on Kaggle, Kaggle job flow fixed for the "previous run still reported complete" race.


## 7. Unseen match, same camera type (2026-10-27)

FIP World Cup Pairs 2025, Kuwait — Coello/Tapia vs Capra/Goñi Lacabe (YouTube sM0GcDOCnFo, 10:00–20:00, 1080p30 with audio). Different tour, season, arena, court colour and broadcast from every training source; same elevated main camera. Calibrated with 6 landmarks (residual 1.6 cm). Analysed on Kaggle (`padel-infer` fetched the video itself; the local uplink was ~50 KB/s). No labels exist, so two label-free checks:

- **Hit precision (eye check, 24 random detected hits):** 17 judgeable from a still (ball visible) → ~14 clear racket contacts, 3 doubtful (ball in the air away from the player) ≈ **0.8**; 7 had no ball detected at that instant (sound-driven) and cannot be judged from a still. Labelled data gives 0.92–0.94 → expect a moderate drop on new venues.
- **Rallies vs the on-screen score:** the scoreboard shows ≈ 24–25 points in the clip. Report: 36 rallies (+44 %). Cause: broadcast replays and close-ups (34 cuts; main court view only 58 % of the time). Ignoring hits outside the main court view is now in the pipeline (no effect on fixed facility cameras, which never cut away). An offline estimate predicted 32; the full re-run on Kaggle gives **35 rallies / 186 hits (+40 %)**: it removed 9 hits but only one rally. The remaining surplus is mostly long rallies split in two by a pause and clusters 5–6 s apart (e.g. 383 / 389 / 395 / 400 s) during replays that keep the court in view.
- **Dead-ball rule (between-point touches):** most surplus "rallies" are 1–3 slow touches by one pair (the ball being knocked back to the server). Rallies of ≤ 3 hits all by one team are now dropped, with their hits. Applied to the stored hit lists: Kuwait **35 → 27 rallies** (scoreboard ≈ 24–25, my reading — not a label), labelled PT100 women's final **75 → 75** (nothing lost), low club clip 32 → 18 (team attribution there is unreliable because near players' feet leave the frame; not the target camera). Thresholds ≤ 1 / 2 / 3 / 4 hits gave Kuwait 30 / 29 / 27 / 27. **Not yet measured against point labels** — needs the Kuwait points marked in the review screen. A 5 s rally gap would give 28 (+12 %) here but cost −18 % on the labelled men's final, so the gap stays 4 s.
- 195 shots ≈ 8 per point (plausible for pro padel); stroke mix 66 forehand / 53 backhand / 39 smash / 28 serve / 9 other.


## 8. Rally segmentation on labelled single-point clips (Decorte, 2026-10-27)

Kernel `padel-rallytest` (CPU). Each of the 99 Decorte clips (11 WPT 2023 tournaments) is exactly one point, so the truth is 1 rally per clip. Hits predicted out-of-fold by tournament (fused audio+video GBM with modality dropout, as in production: 2,390 predicted vs 2,377 true hits), grouped with `engine.rallies` (4 s gap + scene cuts; the serve rule is off because stroke types were not computed here).

| Hits used | Rallies (truth 99) | Exactly 1 | Split in 2+ | Point lost |
| --- | --- | --- | --- | --- |
| true hits | 101 (+2 %) | 97 | 2 | 0 |
| true hits + dead-ball rule | 101 (+2 %) | 97 | 2 | 0 |
| **predicted hits** | 105 (+6 %) | 94 | 5 | 0 |
| **predicted hits + dead-ball rule** | **101 (+2 %)** | **95** | 3 | **1** |

- Splitting long rallies is a small error (+2 % from the gap rule itself, +6 % with detected hits): one true 4.3 s gap, one false scene cut, the rest from missed hits leaving > 4 s gaps.
- The dead-ball rule removes 4 of the 6 spurious rallies and wrongly deletes one real 4-hit point (VIGO_16: its detected hits were all credited to one team).
- What this test cannot measure: touches *between* points (the clips are cut to the point), which is the main surplus on full broadcasts (§7). That still needs labelled point boundaries on a full match.
