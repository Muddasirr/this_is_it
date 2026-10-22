# CourtLens test corpus

| File | Role |
| ---- | ---- |
| `test_video_2.mp4` | **Primary** facility CCTV / product gate |
| `08fd33_4.mp4` | DFL-style 30s clip (broadcast stress) |
| `dfl_*.mp4` | Sample from [DFL 460×30s](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv) (~312MB, 18 clips / 5 matches + test). Many `dfl_extra_test (N).mp4` files are byte-identical repeats — `run_testing_videos.py` skips duplicates (64 files → 49 unique) |
| Portugal / Bangladesh / Spain clips | Broadcast stress only — not product UX target |

Videos stay local (gitignored). Do not commit `.mp4` files.

Labels CSV for the DFL sample: `data/dfl_bundesliga/train.csv`

## External video datasets

| Source | Role | Notes |
| ------ | ---- | ----- |
| [DFL Bundesliga 460 MP4 + CSV](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv) | Broadcast / football stress (tracking, kit split) | Full set ~10GB, attached directly to the Kaggle kernel. The CSV is a file list — **no event labels**. Local sample is `dfl_*.mp4`. |
| `muddasirr/courtlens-test-video-2` | Our Kaggle upload of the local corpus | Used by GPU kernels |

## Live UI (local)

Facility **CourtLens Corpus** — login `corpus179@example.com` / `Testpass123!` → http://localhost:5174

Reports: `data/eval/corpus_ui_reports.json`

## Court formats

Set `COURT_FORMAT=5v5|6v6|7v7` for expected roster size. Product supports all three.

## Commands

```bash
# Local smoke
python scripts/run_testing_videos.py --list
python scripts/run_testing_videos.py --smoke --match test_video_2
python scripts/run_testing_videos.py --smoke --match dfl_

# Full GPU corpus (Kaggle T4) — real backend pipeline, see EVAL.md
python scripts/kaggle/build_kaggle.py --push
kaggle kernels output muddasirr/courtlens-gpu-eval -p data/kaggle_out/gpu_eval

# Import Kaggle/offline outs into UI
python scripts/corpus_to_ui.py
```

Outputs: `data/corpus_out/<slug>/` (local), `data/kaggle_out/gpu_eval/eval/<slug>/` (Kaggle)
