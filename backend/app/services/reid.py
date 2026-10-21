"""OSNet appearance ReID for track fragment merge.

Used after BoT-SORT/ByteTrack: average embeddings per track, merge fragments
that are visually the same person and do not co-exist on court.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ENCODER = None
_ENCODER_DEVICE = "cpu"

# Cosine sim to treat two tracklets as the same person
REID_MERGE_SIM = 0.72
REID_MAX_OVERLAP_MS = 800


def reid_enabled() -> bool:
    from app.core.config import get_settings

    return bool(getattr(get_settings(), "vision_reid", True))


def _load_encoder(device: str = "cpu"):
    global _ENCODER, _ENCODER_DEVICE
    if _ENCODER is not None and _ENCODER_DEVICE == device:
        return _ENCODER
    import torch
    from torchreid import models

    name = "osnet_x0_25"
    try:
        from app.core.config import get_settings

        name = getattr(get_settings(), "vision_reid_model", name) or name
    except Exception:
        pass
    logger.info("Loading ReID encoder %s on %s", name, device)
    m = models.build_model(name=name, num_classes=1000, pretrained=True)
    m.eval()
    m.to(device)
    _ENCODER = m
    _ENCODER_DEVICE = device
    return m


def embed_bgr_crops(crops: list[Any], device: str | None = None) -> list[np.ndarray | None]:
    """Return L2-normalized embeddings for BGR crops (OpenCV). Empty → None."""
    if not crops:
        return []
    import cv2
    import torch

    if device is None:
        from app.core.config import get_settings

        device = "cpu" if get_settings().vision_device in {"cpu", ""} else str(get_settings().vision_device)
        if device == "0":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = _load_encoder(device)
    tensors = []
    valid_idx = []
    for i, crop in enumerate(crops):
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        # ImageNet-ish normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std
        tensors.append(t)
        valid_idx.append(i)
    out: list[np.ndarray | None] = [None] * len(crops)
    if not tensors:
        return out
    batch = torch.stack(tensors, 0).to(device)
    with torch.no_grad():
        feats = model(batch)
    feats = feats.detach().cpu().numpy().astype(np.float32)
    for j, i in enumerate(valid_idx):
        v = feats[j]
        n = float(np.linalg.norm(v) + 1e-12)
        out[i] = (v / n).astype(np.float32)
    return out


def embed_frame_boxes(frame_bgr, boxes_xyxy: list[list[float]], device: str | None = None) -> list[np.ndarray | None]:
    crops = []
    h, w = frame_bgr.shape[:2]
    for xyxy in boxes_xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crops.append(frame_bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None)
    return embed_bgr_crops(crops, device=device)


def ema_update(prev: np.ndarray | None, new: np.ndarray | None, alpha: float = 0.9) -> np.ndarray | None:
    if new is None:
        return prev
    if prev is None:
        return new
    m = alpha * prev + (1.0 - alpha) * new
    n = float(np.linalg.norm(m) + 1e-12)
    return (m / n).astype(np.float32)


def cosine(a: list[float] | np.ndarray | None, b: list[float] | np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def merge_tracks_by_reid(cluster_meta: list[dict]) -> list[dict]:
    """Merge cluster metas whose mean ReID embeddings match and time-overlap is low."""
    if len(cluster_meta) < 2:
        return cluster_meta

    def _overlap_ms(a: list, b: list) -> int:
        total = 0
        for ta in a:
            for tb in b:
                start = max(ta.start_ms, tb.start_ms)
                end = min(ta.end_ms, tb.end_ms)
                if end > start:
                    total += end - start
        return total

    metas = [dict(m, tracks=list(m["tracks"])) for m in cluster_meta]
    changed = True
    while changed:
        changed = False
        metas.sort(key=lambda m: -int(m.get("duration") or 0))
        i = 0
        while i < len(metas):
            j = i + 1
            while j < len(metas):
                e1 = metas[i].get("reid_emb")
                e2 = metas[j].get("reid_emb")
                if e1 is None or e2 is None:
                    j += 1
                    continue
                sim = cosine(e1, e2)
                overlap = _overlap_ms(metas[i]["tracks"], metas[j]["tracks"])
                if sim >= REID_MERGE_SIM and overlap <= REID_MAX_OVERLAP_MS:
                    metas[i]["tracks"].extend(metas[j]["tracks"])
                    # refresh duration / emb average
                    metas[i]["duration"] = sum(t.end_ms - t.start_ms for t in metas[i]["tracks"])
                    # average embeddings of merged
                    embs = []
                    for t in metas[i]["tracks"]:
                        e = (t.features or {}).get("reid_emb")
                        if e:
                            embs.append(np.asarray(e, dtype=np.float32))
                    if embs:
                        m = np.mean(embs, axis=0)
                        m = m / (np.linalg.norm(m) + 1e-12)
                        metas[i]["reid_emb"] = m.tolist()
                    metas.pop(j)
                    changed = True
                    continue
                j += 1
            i += 1
    return metas
