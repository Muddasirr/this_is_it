"""Visual check of cross-segment identity linking: rows = segments (cuts / end changes),
columns = linked persons A1 A2 B1 B2. A correct link shows the same person down each column.

  python scripts/identity_sheet.py --match F --frames 20100 --source oracle|heldout --out sheet.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from padel_analyzer.engine import court as C  # noqa: E402
from padel_analyzer.engine import pipeline  # noqa: E402
from padel_analyzer.engine import players as P  # noqa: E402
from padel_analyzer.eval import pt100  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="F")
    ap.add_argument("--frames", type=int, default=20100)
    ap.add_argument("--source", default="oracle")
    ap.add_argument("--video", default=None)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "identity_sheet.jpg"))
    a = ap.parse_args()
    import train_events as T

    g = pt100.load(a.match)
    n = min(a.frames, g.n)
    det = T.oracle_dets(g, n) if a.source == "oracle" else T.heldout_dets(g, n)
    video = a.video or str(ROOT / "data" / "pt100" / "v1" / f"{pt100.MATCHES[a.match]}.mp4")
    hists = pipeline.sample_hists(video, det)
    tr = P.track(det, C.Calibration(g.H, (1920, 1080)), hists)
    print("segments", len(tr.segments))
    cap = cv2.VideoCapture(video)
    th = 120
    rows = []
    for a0, b0 in tr.segments[: a.rows]:
        f = (a0 + b0) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        cells = []
        for person in range(4):
            slot = int(np.where(tr.person[f] == person)[0][0])
            box = tr.box[f, slot]
            cell = np.full((th, th // 2 + 20, 3), 40, np.uint8)
            if not np.isnan(box).any():
                x0, y0, x1, y1 = box.astype(int)
                crop = img[max(0, y0):y1, max(0, x0):x1]
                if crop.size:
                    s = th / crop.shape[0]
                    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), th))[:, : th // 2 + 20]
                    cell[:, : crop.shape[1]] = crop
            cells.append(cell)
        row = np.hstack([np.full((th, 70, 3), 255, np.uint8)] + cells)
        cv2.putText(row, f"{a0 // 30}s", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rows.append(row)
    head = np.full((30, rows[0].shape[1], 3), 255, np.uint8)
    for i, t in enumerate(["A1", "A2", "B1", "B2"]):
        cv2.putText(head, t, (80 + i * (th // 2 + 20), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(a.out, np.vstack([head] + rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
