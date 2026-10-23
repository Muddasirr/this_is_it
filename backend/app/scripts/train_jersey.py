"""Train the jersey-number classifier on SoccerNet GSR player crops (run on a GPU).

Off-the-shelf OCR fails on shirts (SN-GSR valid, 94 player tracks: Tesseract
7 %, TrOCR 1 % track accuracy). SoccerNet jersey-challenge entries train on
jersey crops with a legibility filter; this does the same, small — see
app/services/jersey_cls.py for the model (ResNet-18 + legible/tens/units heads).

Data: Hugging Face SoccerNet/SN-GSR-2024 train split (public), read from the
remote zip. Boxes with a `jersey` label → legible + digits; player boxes
without one → not legible.

  python -m app.scripts.train_jersey --out /kaggle/working/jersey_resnet18.pt --sequences 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from app.services.jersey_cls import CROP_H, CROP_W, JerseyNet, to_tensor, upper_body

GSR_URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2024/resolve/main/{split}.zip"


def collect(split: str, sequences: int, frame_step: int, per_seq: int, seed: int = 0):
    """(crops uint8 [N, H, W, 3] BGR, labels [N, 3] = legible, tens, units)."""
    import cv2
    from remotezip import RemoteZip

    rng = random.Random(seed)
    xs, ys = [], []
    with RemoteZip(GSR_URL.format(split=split)) as z:
        seqs = sorted({n.split("/")[0] for n in z.namelist() if n.endswith("Labels-GameState.json")})
        rng.shuffle(seqs)
        for seq in seqs[:sequences]:
            labels = json.loads(z.read(f"{seq}/Labels-GameState.json"))
            files = {img["image_id"]: img["file_name"] for img in labels["images"]}
            pos, neg = [], []
            for a in labels["annotations"]:
                attr = a.get("attributes") or {}
                if attr.get("role") not in {"player", "goalkeeper"} or int(str(a["image_id"])[-6:]) % frame_step:
                    continue
                b = a["bbox_image"]
                item = (files[a["image_id"]], [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], attr.get("jersey"))
                (pos if attr.get("jersey") else neg).append(item)
            rng.shuffle(pos)
            rng.shuffle(neg)
            take = pos[: per_seq // 2] + neg[: per_seq // 2]  # balanced legible / not
            frames = {}
            for fname, xyxy, jersey in take:
                if fname not in frames:
                    frames[fname] = cv2.imdecode(np.frombuffer(z.read(f"{seq}/img1/{fname}"), np.uint8), cv2.IMREAD_COLOR)
                crop = upper_body(frames[fname], xyxy)
                if crop is None:
                    continue
                if jersey:
                    n = int(jersey)
                    ys.append((1, n // 10 if n >= 10 else 0, n % 10))
                else:
                    ys.append((0, 0, 0))
                xs.append(crop)
            print(f"{seq}: {len(xs)} crops so far", flush=True)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)



def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sequences", type=int, default=40)
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--per-seq", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    t0 = time.time()
    X, Y = collect("train", args.sequences, args.frame_step, args.per_seq)
    print(f"collected {len(X)} crops ({int(Y[:, 0].sum())} legible) in {time.time() - t0:.0f}s", flush=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    net = JerseyNet.build().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    idx = np.arange(len(X))
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        net.train()
        losses = []
        for i in range(0, len(idx), 128):
            b = idx[i : i + 128]
            x = to_tensor(X[b]).to(dev)
            if np.random.rand() < 0.5:  # brightness / contrast jitter
                x = x * float(np.random.uniform(0.8, 1.2)) + float(np.random.uniform(-0.2, 0.2))
            y = torch.from_numpy(Y[b]).to(dev)
            lg, tn, un = net(x)
            legible = y[:, 0] == 1
            loss = F.cross_entropy(lg, y[:, 0])
            if legible.any():
                loss = loss + F.cross_entropy(tn[legible], y[legible, 1]) + F.cross_entropy(un[legible], y[legible, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "crop": [CROP_H, CROP_W], "arch": "resnet18-3head"}, args.out)
    info = {"crops": len(X), "legible": int(Y[:, 0].sum()), "epochs": args.epochs, "seconds": round(time.time() - t0)}
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    main()
