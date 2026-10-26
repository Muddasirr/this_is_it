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
