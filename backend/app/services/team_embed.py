"""Team split per track: torso jersey colour + SigLIP torso embedding → KMeans.

Clustering follows the roboflow/sports TeamClassifier (embed → reduce → KMeans),
but the measured primary cue is torso jersey colour (Lab, grass/skin/white
masked; median of up to 6 crops per track). On blurry facility CCTV, 12
hand-labelled players: whole-body SigLIP 7/12, torso SigLIP 9/12, torso
colour 11/12, colour + 0.25 × SigLIP 11/12. SigLIP stays as a secondary cue
(EMB_WEIGHT) for kits that share a colour but differ in pattern.

Kit X: with k=3, a third cluster holding ≤ X_MAX_FRAC of the weight is the
refs / goalkeepers (they wear a third colour); otherwise k=2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip-base-patch16-224"
BATCH = 32
X_MAX_FRAC = 0.2
_MODEL: tuple[Any, Any, str] | None = None


def _load(device: str):
    global _MODEL
    if _MODEL is not None and _MODEL[2] == device:
        return _MODEL
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    logger.info("Loading %s on %s", MODEL_ID, device)
    model = SiglipVisionModel.from_pretrained(MODEL_ID).to(device).eval()
    # Image processor only: AutoProcessor also loads the text tokenizer (needs sentencepiece)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    torch.set_grad_enabled(False)
    _MODEL = (model, processor, device)
    return _MODEL


def embed_crops(crops_bgr: list[np.ndarray], device: str = "cpu") -> np.ndarray | None:
    """L2-normalised SigLIP embeddings [N, D] for BGR crops, or None if unavailable."""
    if not crops_bgr:
        return None
    try:
        import torch

        from app.services.reid import torch_device

        dev = torch_device(device)
        model, processor, _ = _load(dev)
    except Exception as exc:  # noqa: BLE001 — optional dependency / offline model
        logger.info("Team embeddings unavailable (%s) — colour kit split will be used", exc)
        return None
    out = []
    for i in range(0, len(crops_bgr), BATCH):
        # BGR → RGB as a contiguous copy: torch rejects the negative-stride view [:, :, ::-1]
        batch = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops_bgr[i : i + BATCH]]
        inputs = processor(images=batch, return_tensors="pt").to(dev)
        with torch.no_grad():
            emb = model(**inputs).pooler_output.float().cpu().numpy()
        out.append(emb)
    embs = np.concatenate(out)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)


def _reduce(x: np.ndarray, dims: int = 3) -> np.ndarray:
    if x.shape[0] <= dims + 2:
        return x
    try:
        import umap

        n_neighbors = max(2, min(15, x.shape[0] - 1))
        return umap.UMAP(n_components=dims, n_neighbors=n_neighbors, random_state=0).fit_transform(x)
    except Exception:  # noqa: BLE001 — umap-learn optional; PCA keeps the same shape
        xc = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        return xc @ vt[:dims].T


EMB_WEIGHT = 0.25


def split_teams(embs: list[list[float]], weights: list[float] | None = None, reduce: bool = True) -> list[str]:
    """Kit letter (A / B / X) per feature row. A = the larger team by weight."""
    from app.services.team_assigner import _kmeans

    x = np.asarray(embs, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return []
    if n == 1:
        return ["A"]
    w = np.asarray(weights if weights is not None else [1.0] * n, dtype=np.float64)
    z = _reduce(x) if reduce else x

    labels = None
    if n >= 6:
        lab3, cent = _kmeans(z, 3)
        mass = np.array([w[lab3 == k].sum() for k in range(3)])
        order = np.argsort(-mass)
        a, b, x = (cent[int(k)] for k in order)
        # A real third kit sits far from both teams; a split-off sliver of one team does not
        separated = min(np.linalg.norm(x - a), np.linalg.norm(x - b)) >= 0.5 * np.linalg.norm(a - b)
        if mass[order[2]] <= X_MAX_FRAC * mass.sum() and separated:
            names = {int(order[0]): "A", int(order[1]): "B", int(order[2]): "X"}
            labels = [names[int(k)] for k in lab3]
    if labels is None:
        lab2, _ = _kmeans(z, 2)
        mass = np.array([w[lab2 == k].sum() for k in range(2)])
        big = int(np.argmax(mass))
        labels = ["A" if int(k) == big else "B" for k in lab2]
    return labels


def assign_kits_by_embedding(metas: list[dict], min_coverage: float = 0.7) -> bool:
    """Set meta['kit'] from per-track team embeddings; False if too few tracks have them.

    Refs already flagged by the detector (class referee) stay Kit X.
    """
    rows = []  # (meta index, colour or None, embedding or None, weight)
    for i, m in enumerate(metas):
        if m.get("kit") == "X" and m.get("class_name") == "referee":
            continue
        cols, cw, embs, ew = [], [], [], []
        for t in m.get("tracks") or []:
            feats = getattr(t, "features", None) or {}
            w = max(1, int(getattr(t, "end_ms", 0)) - int(getattr(t, "start_ms", 0)))
            if feats.get("team_colour"):
                cols.append(np.asarray(feats["team_colour"], dtype=np.float64))
                cw.append(w)
            if feats.get("team_emb"):
                embs.append(np.asarray(feats["team_emb"], dtype=np.float64))
                ew.append(w)
        col = np.average(np.stack(cols), axis=0, weights=cw) if cols else None
        emb = None
        if embs:
            emb = np.average(np.stack(embs), axis=0, weights=ew)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
        if col is not None or emb is not None:
            rows.append((i, col, emb, float(sum(cw) or sum(ew))))
    candidates = [m for m in metas if not (m.get("kit") == "X" and m.get("class_name") == "referee")]
    if not candidates or len(rows) < max(2, min_coverage * len(candidates)):
        return False
    kits = split_teams(_team_features(rows), [r[3] for r in rows], reduce=False)
    # Clusters without embeddings keep their colour kit (set by the caller first),
    # relabelled so colour "A" means the same team as embedding "A".
    colour = {r[0]: metas[r[0]].get("kit") for r in rows}
    same = sum(1 for r, k in zip(rows, kits) if colour[r[0]] == k and k in {"A", "B"})
    swap = sum(1 for r, k in zip(rows, kits) if {colour[r[0]], k} == {"A", "B"})
    flip = {"A": "B", "B": "A"} if swap > same else {}
    have = {r[0] for r in rows}
    for i, m in enumerate(metas):
        if i not in have and m.get("kit") in flip:
            m["kit"] = flip[m["kit"]]
    for r, kit in zip(rows, kits):
        metas[r[0]]["kit"] = kit
        metas[r[0]]["kit_method"] = "colour+siglip" if r[1] is not None and r[2] is not None else (
            "colour" if r[1] is not None else "siglip")
    return True


def _per_dim(x: np.ndarray) -> np.ndarray:
    """Colour block: every Lab channel matters — CCTV labels: 11/12 vs 8–10/12 with one global scale."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def _global(x: np.ndarray) -> np.ndarray:
    """Embedding block: one scale factor, so trailing near-noise reduced dims stay small."""
    c = x - x.mean(axis=0)
    return c / (float(np.sqrt((c**2).sum(axis=1).mean())) + 1e-6)


def _team_features(rows: list[tuple]) -> list[list[float]]:
    """Standardised torso colour (primary) + EMB_WEIGHT × reduced SigLIP (secondary)."""
    n = len(rows)
    parts = []
    if all(r[1] is not None for r in rows):
        parts.append(_per_dim(np.stack([r[1] for r in rows])))
    if all(r[2] is not None for r in rows):
        z = np.asarray(_reduce(np.stack([r[2] for r in rows]), 3), dtype=np.float64)
        parts.append((EMB_WEIGHT if parts else 1.0) * _global(z) * (np.sqrt(3) if parts else 1.0))
    if not parts:  # mixed coverage: colour where present, else zeros (neutral)
        dim = len(next(r[1] for r in rows if r[1] is not None))
        parts.append(_per_dim(np.stack([r[1] if r[1] is not None else np.zeros(dim) for r in rows])))
    return np.hstack(parts).reshape(n, -1).tolist()
