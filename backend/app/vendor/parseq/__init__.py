"""PARSeq scene-text recogniser (Bautista & Atienza, ECCV 2022) — vendored for inference.

Source: https://github.com/baudm/parseq @ 1902db043c029a7e03a3818c616c06600af574be
(Apache-2.0, LICENSE here). See VENDORED.md for what was copied and changed.
Weights: release v1.0.0 `parseq-bb5792a6.pt`, verified by SHA-256 before loading,
loaded with torch.load(weights_only=True) (no pickled code).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PARSEQ_SHA256 = "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
# configs/main.yaml + configs/model/parseq.yaml at the pinned commit
CONFIG = dict(max_label_length=25, img_size=(32, 128), patch_size=(4, 8), embed_dim=384, enc_num_heads=6,
              enc_mlp_ratio=4, enc_depth=12, dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1, decode_ar=True,
              refine_iters=1, dropout=0.1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(weights: str | Path, device: str = "cpu"):
    """(model, tokenizer) in eval mode; raises if the weights file is not the pinned release."""
    import torch

    from .model import PARSeq
    from .tokenizer import Tokenizer

    path = Path(weights)
    if _sha256(path) != PARSEQ_SHA256:
        raise ValueError(f"{path}: SHA-256 does not match the pinned PARSeq release")
    tokenizer = Tokenizer(CHARSET)
    model = PARSeq(len(tokenizer), **CONFIG)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), tokenizer


def preprocess(crops_bgr):
    """BGR crops → [N, 3, 32, 128] tensor (bicubic resize, normalised to [-1, 1]) as in the repo's SceneTextDataModule."""
    import cv2
    import numpy as np
    import torch

    h, w = CONFIG["img_size"]
    arr = np.stack([cv2.resize(c[:, :, ::-1], (w, h), interpolation=cv2.INTER_CUBIC) for c in crops_bgr]).astype(np.float32)
    t = torch.from_numpy(arr / 255.0).permute(0, 3, 1, 2)
    return (t - 0.5) / 0.5
