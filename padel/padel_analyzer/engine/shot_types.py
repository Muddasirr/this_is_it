"""Padel-specific shot taxonomy per hit from a YOLO detector trained on padelcnn/padel-quret (Roboflow
Universe, CC BY 4.0, 9.9k frames labelled with the shot each player is playing). Adds what PadelTracker100
lacks: bandeja, lob, forehand / backhand volley, off-the-wall, contrapared. Estimated; run on the frames
around each hit and matched to the hitter's box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "models" / "shot_types.pt"
NAMES = {
    "Bandeja": "bandeja", "Remate": "smash", "Globo": "lob", "Golpe de derecha": "forehand",
    "Golpe de reves": "backhand", "Volea de derecha": "forehand volley", "Volea de reves": "backhand volley",
    "Salida de pared": "off the wall", "Contrapared": "contrapared", "Saque": "serve",
}
OFFSETS = (-3, -1, 0, 1, 3)


def available() -> bool:
    return MODEL.exists()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify(video: str | Path, hits, tracks, step: int = 1, device: str | None = None) -> list[tuple[str | None, float]]:
    """(padel_type, confidence) per hit; (None, 0) when nothing overlapping the hitter was detected."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    names = model.names
    cap = cv2.VideoCapture(str(video))
    out: list[tuple[str | None, float]] = []
    for h in hits:
        votes: dict[str, float] = {}
        for o in OFFSETS:
            f = h.frame + o
            if f < 0 or f >= len(tracks.box):
                continue
            hb = tracks.box[f, h.slot]
            if np.isnan(hb).any():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f * step)
            ok, img = cap.read()
            if not ok:
                continue
            r = model.predict(img, imgsz=640, conf=0.25, verbose=False, device=device)[0]
            for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                label = NAMES.get(names[int(c)])
                if label and _iou(box, hb) > 0.2:
                    votes[label] = votes.get(label, 0.0) + float(s)
        if votes:
            best = max(votes, key=votes.get)
            out.append((best, round(votes[best] / sum(votes.values()), 3)))
        else:
            out.append((None, 0.0))
    cap.release()
    return out
