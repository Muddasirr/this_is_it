"""Experiment: temporal CNN over the hitter's pose sequence vs the GBM stroke classifier.

Cross-match on oracle perception (GT pose + ball): train F -> test M and M -> F.
Mirror augmentation (flip x, swap left/right joints) teaches handedness invariance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_events as T  # noqa: E402

from padel_analyzer.engine import shots as S  # noqa: E402

WIN = (-15, 10)
KP = list(range(5, 17))  # shoulders .. ankles (head points are unrefined in PT100)
SWAP = {5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}


def seq(d: dict) -> tuple[np.ndarray, np.ndarray]:
    _, y, hits = T.shot_xy(d)
    tr, ball = d["tracks"], d["ball"]
    n = len(tr.kpt)
    X = np.zeros((len(hits), WIN[1] - WIN[0], len(KP) * 2 + 3), np.float32)
    for i, h in enumerate(hits):
        flip = -1.0 if h.slot < 2 else 1.0  # far side faces the camera
        k0 = tr.kpt[h.frame, h.slot]
        hip0 = (k0[11, :2] + k0[12, :2]) / 2
        torso = max(np.linalg.norm((k0[5, :2] + k0[6, :2]) / 2 - hip0), 1.0)
        for j, o in enumerate(range(*WIN)):
            f = min(max(h.frame + o, 0), n - 1)
            k = tr.kpt[f, h.slot]
            idx = [SWAP[q] if flip < 0 else q for q in KP]  # far players: image-left is their right
            p = (k[idx, :2] - hip0) / torso
            p[:, 0] *= flip
            b = (ball.xy[f] - hip0) / torso
            X[i, j, : 2 * len(KP)] = np.nan_to_num(p.ravel())
            X[i, j, -3:] = [np.nan_to_num(b[0] * flip), np.nan_to_num(b[1]), float(np.isnan(b).any())]
    return X, y


def mirror(X: np.ndarray) -> np.ndarray:
    M = X.copy()
    P = M[:, :, : 2 * len(KP)].reshape(len(X), X.shape[1], len(KP), 2)
    perm = [KP.index(SWAP[q]) for q in KP]
    P = P[:, :, perm]
    P[..., 0] *= -1
    M[:, :, : 2 * len(KP)] = P.reshape(len(X), X.shape[1], -1)
    M[:, :, -3] *= -1
    return M


class Net(nn.Module):
    def __init__(self, c_in: int, n_cls: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c_in, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.f(x.transpose(1, 2))


def fit_predict(Xtr, ytr, Xte, seed=0, epochs=120, mirror_aug=True) -> np.ndarray:
    torch.manual_seed(seed)
    if mirror_aug:
        Xtr, ytr = np.concatenate([Xtr, mirror(Xtr)]), np.concatenate([ytr, ytr])
    net = Net(Xtr.shape[2], len(S.CLASSES))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    w = torch.tensor(1.0 / np.maximum(np.bincount(ytr, minlength=len(S.CLASSES)), 1), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(S.CLASSES))
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]
            xb = xt[b] + 0.02 * torch.randn_like(xt[b])
            opt.zero_grad()
            lossf(net(xb), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.tensor(Xte)), 1).numpy()
        p += torch.softmax(net(torch.tensor(mirror(Xte))), 1).numpy()  # test-time mirror averaging
    return p


def main() -> None:
    data = {k: T.prepare(k, "oracle") for k in "FM"}
    sq = {k: seq(d) for k, d in data.items()}
    for tr, te in (("F", "M"), ("M", "F")):
        Xtr, ytr = sq[tr]
        Xte, yte = sq[te]
        P = sum(fit_predict(Xtr, ytr, Xte, seed=s) for s in range(3))
        pred = P.argmax(1)
        print(f"test {te}: seq-CNN acc {np.mean(pred == yte):.4f} macroF1 {T.macro_f1(yte, pred):.4f}")


if __name__ == "__main__":
    main()
