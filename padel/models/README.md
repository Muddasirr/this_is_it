# Models

Trained weights (gitignored). Pull from the Kaggle GPU kernel output:

- `ball.pt` - YOLO11s ball detector @1536 (PadelTracker100, both finals)
- `pose.pt` - YOLO11m-pose players @1280
- `hit_lgbm.txt/.json` - hit classifier + threshold (`scripts/train_events.py --save`)
- `shot_lgbm.txt` - stroke classifier
