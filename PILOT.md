# CourtLens — pilot product plan (1 futsal court)

## Locked decisions (from product owner)

| Decision | Choice |
| -------- | ------ |
| Cameras | **1–4** supported (A–D). Pilot court runs Camera A; production courts add B–D |
| Hosting | **Cloud** (facility uploads match video) |
| Quality bar | **Good enough for a satisfied pilot client** + staff can correct |
| Stats | Passes, shots, goals, possession, heatmaps, distance, touches, identity, shareable report |
| Identity | **Kit color + jersey number** auto, staff can override |
| Customers | **1 pilot court** first |
| Compute | CPU worker for the pilot; Kaggle T4 runs the same pipeline for eval / batch (`EVAL.md`) |

## What we ship for the pilot

1. Facility login → create match → upload Camera A (optional B–D)
2. Cloud processing: detect/track → color+number identity → events → stats
3. Match centre: G/S/P, possession, touches, distance, heatmaps, key moments review
4. Publish → **public share link** courts can send to players/coaches
5. Staff can rename players / reassign events

## Honest quality notes for the pilot court

- Ball-heavy stats (shots/goals) need visible ball; wide CCTV may under-count
- Shirt numbers are only proposed by the trained jersey classifier when configured (generic OCR was right for 7 % of players on SoccerNet); otherwise kit colour + staff rename
- All auto stats stay **estimated** until staff confirm

## Free GPU options (training / faster offline runs)

These are **not** a free always-on production GPU for your cloud API. Use them to train detectors or batch-process offline.

**Wired today:** Kaggle. `python scripts/kaggle/build_kaggle.py --push` uploads the backend and runs kernel `muddasirr/courtlens-gpu-eval` (test_video_2, DFL clips, synthetic 3-camera check) with the exact product code.

| Option | Rough free quota | Best for |
| ------ | ---------------- | -------- |
| [Kaggle Notebooks](https://www.kaggle.com/) | ~30 GPU hours/week (phone verify) | Batch process matches, fine-tune YOLO |
| [Google Colab](https://colab.research.google.com/) | Free GPU when available (limits unpublished) | Experiments, short training |
| [Lightning AI](https://lightning.ai/) | Free credits after signup/phone verify | Persistent studio + GPU bursts |
| [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zero-gpu) | Minutes/day on Spaces | Tiny demos only |

**Production path on CPU:** keep `VISION_FRAME_STRIDE=15` (or higher) in the worker. When you have budget, set `VISION_DEVICE=0` on a GPU cloud box (RunPod / Vast / AWS g4dn) for faster matches.

## Pilot success checklist

- [ ] Court can log in and upload a full match
- [ ] Processing completes without engineer help
- [ ] Player board shows identity + POS% / P / heatmap
- [ ] Court can fix wrong names in Identity tab
- [ ] Owner can add staff by email (Staff panel)
- [ ] Share link opens a clean report without login; **Revoke share link** kills it
- [ ] Court says “good enough to use again”

## How the pilot court uses it

1. Sign in at the CourtLens web app (cloud).
2. Open their facility → **Create match**.
3. **Setup & process** → upload Camera A (optional B–D) → check **Camera sync** offsets if more than one camera → **Start processing**.
4. **Match centre** → review stats / key moments; rename players under **Player identity**.
5. **Publish & share report** → copy link → send to coaches/players (`/report/<token>`, no login).

Re-process after code updates: **Retry** on the match (worker must be restarted when Python vision code changes).
