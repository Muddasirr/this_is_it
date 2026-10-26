# Padel analytics — research notes (2026-10)

What exists, what works, and what we take from it. Sources are linked inline; numbers are as reported by the authors unless marked *measured* (ours, see `EVAL.md`).

---

## 1. Datasets

| Dataset | What | Size / licence | Use in PadelLens |
| --- | --- | --- | --- |
| **PadelTracker100** ([Data in Brief 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12926558/), [Zenodo v1 – videos+labels](https://zenodo.org/records/14653706), [Zenodo v2 – labels+homography](https://zenodo.org/records/17020011)) | 2022 World Padel Tour Barcelona finals (women's 45,934 frames, men's 53,953), single main broadcast camera 1920×1080@30, 7.6 m high / 15.5 m behind the glass. Per frame: ball box (last point of motion blur), 4 player boxes + 17 COCO keypoints (ViTPose-L + manual correction), player floor positions in metres (ankle midpoint through a homography), the 3×3 homography itself, shot windows (Serve / Forehand / Backhand / Smash / Dropshot / Other, ~15 frames around each contact). | v1 zip 7.1 GB (both full match videos + labels); v2 labels 70 MB. **CC BY 4.0** | **Primary ground truth** for ball, player detection/pose, court calibration, metres, hit detection, stroke type. Cross-match protocol: train on one final, test on the other. |
| Decorte et al., *Multi-Modal Hit Detection and Positional Analysis in Padel Competitions* ([CVPRW 2024](https://mlanthology.org/cvprw/2024/decorte2024cvprw-multimodal)) | Open padel rally dataset annotated for hits and player–ball interactions; audio hit detector F1 ≈ 0.92, hit→player 83.7 %, hit→team 86.8 %. | rallies, open | Second-tournament test set + audio hit cue (roadmap P2). |
| Padel IMU strokes ([Cartes Domínguez et al.](https://github.com/guillecartes/Padel-Shot-Classification-and-Dataset)) | 2,328 strokes, 12 players, 13 stroke types from a wrist IMU; 1D-CNN 93 %. | sensor only | Taxonomy reference (bandeja / víbora / volleys), not video. |
| PadelVic | Amateur multi-view video + estimated player positions + synthetic video. | small | Amateur-court domain check (roadmap). |
| Roboflow Universe padel sets ([padel-detection-5](https://universe.roboflow.com/padel-rbqpp/padel-detection-5), [Padel_VV](https://universe.roboflow.com/najla02/padel_vv), [padel specific key point](https://universe.roboflow.com/plaimaker/padel-specific-key-point)) | Ball / player / net / fence boxes, some court keypoints, mixed club cameras. | CC BY 4.0 (per page) | Venue diversity for the ball + player detectors (P1, `ROBOFLOW_API_KEY` in `.env`). |

Key property of PadelTracker100 that the eval relies on: the homography is constant per match (fixed camera), and player identities in the labels are per-annotation, not per-track — identity ground truth is built by linking floor positions frame-to-frame (players never teleport; two players of a pair are on the same side of the net).

---

## 2. Reference systems (open source)

| Project | Pipeline | Reported | Notes |
| --- | --- | --- | --- |
| [lmrt0572/padel-analysis](https://github.com/lmrt0572/padel-analysis) | Calibrate → players (Ultralytics) → ball candidates + **sequence-level trajectory choice** → temporal contact net (ball trajectory + gestures + geometry) → surface touched by intersecting the camera ray with the known court surface (floor/glass/mesh) → scoreboard OCR → rallies/points/stats | player F1 0.88, IDF1 0.84; ball within 10 px 80 % (trajectory selection lifted recall 16 → 72 %); contacts 62 % rules → 78 % learned → 83 % ensemble + glass physics; surface 82–83 % end-to-end | AGPL-3.0. Best evidence that **trajectory-level ball selection** and a **learned temporal contact model** beat per-frame rules. Glass bounces at the far end move the ball only a few pixels. |
| [HASHU7/padel](https://github.com/HASHU7/padel) | Fine-tuned pose (players+keypoints), ball detector, top-5 candidates per frame linked over time ("link_strict"), impulse classification (hit near racket / bounce / wall), shot-type classifier + temporal refinement, scoreboard OCR | trained on one PT100 final, tested on the other: players P/R 99/99 %, PCK@0.1 96 %, ball P/R@10 px 88/89 % (F) 90/84 % (M), shot F1 0.91 / 0.88, stroke type 81–82 % (≈87 % without "Other") | Same cross-match protocol we use → **direct comparison point**. Off-the-shelf YOLO11s-pose was clearly worse (kpt mAP 0.47–0.62 vs 0.67). |
| [Joao-M-Silva/padel_analytics](https://github.com/Joao-M-Silva/padel_analytics) | Fixed camera; user clicks **12 court keypoints** → homography; player tracking, pose → stroke class (volley/bandeja/smash), ball hit prediction | position/velocity, heatmaps, ball speed per stroke | Manual one-time calibration is the norm for fixed club cameras. |
| [Padex](https://pypi.org/project/padex/) | YOLO + ByteTrack players, automatic court-line detection + homography, event stream | — | Automatic line-based calibration is feasible on padel (strong white lines, blue/green floor). |
| Decorte et al. (CVPRW 2024) | **Audio** hit detection + pose re-ID + distances to net | hit F1 ≈ 0.92 | Racket impact sound is a strong, cheap hit cue on club footage. |

## 3. Models / methods

**Ball (small, fast, blurred, occluded by glass reflections and players)**
- *Per-frame detector*: YOLO at high input resolution (the PT100 ball is ~8 px at 1920 wide: at 640 input it is 2–3 px and disappears). We train YOLO11s at 1536.
- *Heatmap trackers*: TrackNet ([arXiv 1907.03698](https://ar5iv.labs.arxiv.org/html/1907.03698)), TrackNetV3 (ACM MM, two modules: trajectory prediction with mixup + **rectification by inpainting** the occluded path; 87.7 → 97.5 % accuracy on shuttlecock; MIT), **WASB** ([BMVC 2023](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/310), [code](https://github.com/nttcom/WASB-SBDT): multi-frame HRNet heatmap, tennis F1 95.6, badminton 93.1). RacketVision ([arXiv 2511.17045](https://arxiv.org/pdf/2511.17045)) benchmarks both across racket sports.
- *Temporal selection*: keep top-k candidates per frame and pick the physically consistent path over the sequence (Viterbi / DP) — the single biggest lever in lmrt0572 (16 → 72 % recall).
- Plan: YOLO + top-5 + Viterbi now (fits the existing Ultralytics stack and CPU inference); WASB/TrackNetV3 3-frame heatmap model as the P1 upgrade if recall@10px < 0.85.

**Players**
- YOLO pose (boxes + 17 keypoints in one pass). Fine-tuning on PT100 matters for far-court players (off-the-shelf misses the small far pair; HASHU7 shows the same).
- Floor position = ankle midpoint → homography (same definition as the PT100 metres).
- Tracking: there are exactly 4 players and each pair stays on its side of the net within a point → a constrained 4-slot assignment (Hungarian on metres with constant-velocity prediction) is more robust than generic MOT; teams = net side; appearance (torso colour) only to re-link slots after side changes / broadcast cuts.

**Court calibration**
- Floor plane homography from ≥4 of 13 floor landmarks (corners, service-line ends, service T's, net-line ends/centre). FIP court: 10 × 20 m, service lines 6.95 m from the net, net 0.88 m centre / 0.92 m posts ([FIP Rules of Padel](https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel.pdf)).
- Automatic refinement: white-line mask → distance transform → optimise H so projected court lines land on line pixels (chamfer). Template priors for typical club camera mounts give the initial guess.
- Ball height is not observable from one camera without a physics model; ball metres are only valid at floor contact. Ball "speed" is therefore reported as **shot-to-shot ground speed** (hitter position → next hitter position over flight time), labelled *estimated*.

**Hits, bounces, stroke types**
- Hit = racket contact. Features: change of ball image velocity (direction/speed), ball–wrist distance normalised by player height, wrist speed, ball side relative to the net; learned per-frame classifier (gradient boosting) + peak picking. Hitter = player whose wrist is closest at the peak.
- Stroke type from the hitter's pose window (normalised keypoints around contact), ball height relative to the head and side relative to the body, court zone, rally position (first shot = serve). PT100 classes: serve, forehand, backhand, smash, (dropshot → too rare, folded into "other").
- Padel-specific derived tags (estimated): **volley** (contact without a preceding bounce on the hitter's side, player inside ~4 m of the net), **overhead** (wrist above head: bandeja / víbora / smash family; smash is learned, bandeja vs víbora needs more labels — roadmap), **after-glass** (rebound off the back/side glass before the hit).

**Points / score**
- Broadcast: scoreboard OCR (both reference systems). Club cameras have no scoreboard → rally end + last hitter + review UI; winner/error attribution confirmed by staff/players (same "staff override wins" principle as CourtLens futsal).

## 4. Licensing

Ultralytics (YOLO) is AGPL-3.0, already used by the futsal product; PadelTracker100 and the Roboflow sets are CC BY 4.0 (attribution required in the product docs); TrackNetV3 MIT; WASB check repo licence before vendoring. lmrt0572 is AGPL — ideas only, no code copied.
