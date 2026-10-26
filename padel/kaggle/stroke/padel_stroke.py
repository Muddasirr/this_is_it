"""Padel GPU kernel #3: RGB clip stroke classifier on the hitter crop (sees the racket hand).

Pose-only stroke models cap at ~0.62 on the women's final because pose cannot tell which hand holds
the racket (docs/EVAL.md §2). Here: 16 frames around each PT100 shot, cropped on the hitter,
R(2+1)D-18 (Kinetics-400 pretrained) fine-tuned; cross-match F->M, M->F, then both -> stroke_rgb.pt.

Inputs: padel-pt100-prep (videos + labels), padel-code (padel_analyzer for GT loading).
"""
from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

T_BEFORE, T_AFTER, N_FR, SIZE = 10, 6, 16, 112
OUT = Path("/kaggle/working/out")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        sys.exit(f"missing {pattern}")
    return Path(hits[0])


def setup():
    code = find("/kaggle/input/**/padel_analyzer/__init__.py").parents[1]
    prep = find("/kaggle/input/**/pt100/videos").parent
    repo = Path("/tmp/padel")  # not under /kaggle/working: keeps 600 MB of labels out of the output
    if not repo.exists():
        shutil.copytree(code, repo)
        lab = repo / "data" / "pt100" / "labels"
        lab.mkdir(parents=True)
        for f in (prep / "labels").iterdir():
            shutil.copy(f, lab / f.name)
    sys.path.insert(0, str(repo))
    return repo, prep


def hitter_box(g, f: int) -> np.ndarray | None:
    """Box of the player whose wrist is closest to the ball near frame f (GT pose + GT ball)."""
    best = None
    for df in (0, -1, 1, -2, 2, -3, 3, -5, 5):
        h = f + df
        if 0 <= h < g.n and not np.isnan(g.ball[h, 0]):
            b = g.ball[h]
            boxes, k = g.pbox[h], g.pkpt[h]
            d = []
            for p in range(len(boxes)):
                if np.isnan(boxes[p, 0]):
                    d.append(np.inf)
                    continue
                w = k[p, [9, 10], :2]
                d.append(np.linalg.norm(w - b, axis=1).min())
            best = int(np.argmin(d))
            if np.isfinite(d[best]):
                return g.pbox[f, best] if not np.isnan(g.pbox[f, best, 0]) else boxes[best]
    return None


def clips(g, video: Path, shots, classes) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    X, y, frames_kept = [], [], []
    for s in shots:
        box = hitter_box(g, s.frame)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * 1.6  # room for the racket swing
        a = int(round(cx - side / 2)), int(round(cy - side / 2))
        start = max(0, s.frame - T_BEFORE)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(N_FR):
            ok, fr = cap.read()
            if not ok:
                break
            pad = cv2.copyMakeBorder(fr, 400, 400, 400, 400, cv2.BORDER_CONSTANT)
            c = pad[a[1] + 400:a[1] + 400 + int(side), a[0] + 400:a[0] + 400 + int(side)]
            frames.append(cv2.resize(c, (SIZE, SIZE))[:, :, ::-1])
        if len(frames) < N_FR:
            continue
        X.append(np.stack(frames))
        y.append(classes.index(s.kind))
        frames_kept.append(s.frame)
    return np.asarray(X, np.uint8), np.asarray(y), np.asarray(frames_kept)


def train_eval(Xtr, ytr, Xte, yte, n_cls, epochs=18, seed=0):
    import torch
    import torch.nn as nn
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    torch.manual_seed(seed)
    dev = "cuda"
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    m = m.to(dev)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=dev).view(1, 3, 1, 1, 1)

    def prep(xb, train):
        x = torch.from_numpy(xb).to(dev).float().div(255).permute(0, 4, 1, 2, 3)  # B C T H W
        if train:
            flip = torch.rand(len(x), device=dev) < 0.5  # mirror = other-handed player, same stroke
            x[flip] = x[flip].flip(-1)
            x = x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, 1, device=dev))
        return (x - mean) / std

    cnt = np.bincount(ytr, minlength=n_cls)
    w = torch.tensor(len(ytr) / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=epochs * int(np.ceil(len(Xtr) / 16)))
    scaler = torch.amp.GradScaler()
    for ep in range(epochs):
        m.train()
        perm = np.random.default_rng(seed + ep).permutation(len(Xtr))
        for i in range(0, len(perm), 16):
            b = perm[i:i + 16]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(m(prep(Xtr[b], True)), torch.from_numpy(ytr[b]).to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    m.eval()
    P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(Xte), 32):
            x = prep(Xte[i:i + 32], False)
            p = torch.softmax(m(x).float(), 1) + torch.softmax(m(x.flip(-1)).float(), 1)
            P.append(p.cpu().numpy())
    return m, (np.concatenate(P) if P else np.zeros((0, n_cls)))


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        if tp + fn:
            f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip"], check=False)
    OUT.mkdir(parents=True, exist_ok=True)
    repo, prep = setup()
    from padel_analyzer.eval import pt100

    classes = pt100.SHOT_CLASSES
    data, frames, heldout = {}, {}, {}
    for k in "FM":
        g = pt100.load(k, repo / "data" / "pt100" / "labels")
        X, y, fr = clips(g, prep / "videos" / f"{pt100.MATCHES[k]}.mp4", g.shots, classes)
        data[k] = (X, y)
        frames[k] = fr
        log(k, "clips", X.shape, np.bincount(y, minlength=len(classes)).tolist())
    res = {}
    import torch

    for tr, te in (("F", "M"), ("M", "F")):
        P = None
        for seed in range(2):
            _, p = train_eval(*data[tr], *data[te], len(classes), seed=seed)
            P = p if P is None else P + p
        pred = P.argmax(1)
        y = data[te][1]
        heldout[te] = P / P.sum(1, keepdims=True)
        conf = {classes[c]: {classes[q]: int(((y == c) & (pred == q)).sum()) for q in range(len(classes))}
                for c in range(len(classes))}
        res[f"test_{te}"] = {"acc": round(float((pred == y).mean()), 4),
                             "macro_f1": round(macro_f1(y, pred, len(classes)), 4), "n": int(len(y)),
                             "confusion": conf}
        log("test", te, res[f"test_{te}"]["acc"], res[f"test_{te}"]["macro_f1"])
        (OUT / "stroke_rgb_eval.json").write_text(json.dumps(res, indent=1))
    # held-out probabilities per GT shot frame, for fusing with the pose/context model
    np.savez(OUT / "stroke_rgb_heldout.npz", **{f"P_{k}": heldout[k] for k in heldout},
             **{f"frames_{k}": frames[k] for k in frames}, **{f"y_{k}": data[k][1] for k in data},
             classes=np.array(classes))
    Xa = np.concatenate([data["F"][0], data["M"][0]])
    ya = np.concatenate([data["F"][1], data["M"][1]])
    m, _ = train_eval(Xa, ya, Xa[:1], ya[:1], len(classes))
    torch.save({"state_dict": m.state_dict(), "classes": classes, "size": SIZE, "t_before": T_BEFORE,
                "n_frames": N_FR, "crop_scale": 1.6}, OUT / "stroke_rgb.pt")
    log("done", json.dumps(res)[:1000])


if __name__ == "__main__":
    main()
