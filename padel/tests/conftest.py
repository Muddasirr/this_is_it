import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# PT100 women's final camera (image px -> court metres)
H_PT100 = np.array([[0.026098095342145942, 0.014904213484550795, -19.946805713490495],
                    [0.0004925894191014476, 0.12221633966441776, -39.13767794560289],
                    [2.573098798160646e-05, 0.002983857630570546, 1.0]])


@pytest.fixture
def calib():
    from padel_analyzer.engine import court as C

    return C.Calibration(H_PT100.copy(), (1920, 1080))


def synthetic_detections(calib, n=300, fps=30.0, cut_at=None, seed=0):
    """4 players moving smoothly on their halves, rendered to px boxes + ankle keypoints; a ball
    flying between the pairs; optional hard cut after which every player is somewhere else."""
    from padel_analyzer.engine.detect import Detections

    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    base = np.array([[3.0, 4.0], [7.0, 5.0], [3.0, 16.0], [7.0, 15.0]])
    pos = base[None] + 0.8 * np.stack([np.sin(t[:, None] * [0.7, 1.1, 0.9, 1.3]),
                                       np.cos(t[:, None] * [0.5, 0.8, 1.2, 0.6])], -1)
    if cut_at is not None:
        pos[cut_at:, :, 1] += np.where(pos[cut_at:, :, 1] < 10, 3.0, -3.0)  # camera cut: everyone elsewhere
    box = np.zeros((n, 10, 5), np.float32)
    kpt = np.zeros((n, 10, 17, 3), np.float32)
    for p in range(4):
        feet = calib.to_px(pos[:, p])
        h = 110 + 60 * (pos[:, p, 1] / 20)
        box[:, p] = np.c_[feet[:, 0] - h / 4, feet[:, 1] - h, feet[:, 0] + h / 4, feet[:, 1], np.full(n, 0.9)]
        kpt[:, p, 15, :2] = feet + [-5, 0]
        kpt[:, p, 16, :2] = feet + [5, 0]
        kpt[:, p, :, 2] = 0.9
        for j in range(15):
            kpt[:, p, j, 0] = feet[:, 0] + rng.normal(0, 3)
            kpt[:, p, j, 1] = feet[:, 1] - h * (1 - j / 17)
    ball = np.zeros((n, 5, 5), np.float32)
    bx = 960 + 300 * np.sin(t * 2)
    by = 500 + 250 * np.sin(t * 3)
    ball[:, 0] = np.c_[bx - 4, by - 4, bx + 4, by + 4, np.full(n, 0.8)]
    ball[::7, 0, 4] = 0  # dropouts
    ball[:, 1] = [100, 900, 108, 908, 0.7]  # static look-alike
    return Detections(fps, (1920, 1080), ball, box, kpt, np.ones(n, bool), np.zeros(n, bool)), pos
