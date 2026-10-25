"""Zero-shot SigLIP on each track's full-body crop: player filter + kit colour profile.

On an uncalibrated phone/CCTV view a third of the person tracks are spectators,
coaches and staff. Left in, the team KMeans splits "players vs bystanders"
instead of the two kits (futsal phone clip, 60 hand-labelled tracks: red team
21/25 and black team 10/11 both landed in Kit A, bystanders 18/20 in Kit B).

score = cos(crop, "player" prompts) − cos(crop, "bystander" prompts); > 0 ⇒ player.
Same clip: 92 % correct at the untuned threshold 0 on full-body crops (torso
crops 42 %). Calibrated cameras already drop off-court people geometrically;
this covers every camera.

colour profile = softmax over "a person wearing a {colour} shirt" for a fixed
palette. As the team-split feature it separated the clip's red and black teams
on 35/36 labelled players (97 %); torso Lab colour 50 %, torso SigLIP 61 % —
the Lab picker chose the black kit's pink numbers as its colour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "google/siglip-base-patch16-224"
BATCH = 32
PLAYER_PROMPTS = [
    "a photo of a futsal player wearing a sports jersey and shorts",
    "a photo of a football player in team kit",
]
OTHER_PROMPTS = [
    "a photo of a spectator in street clothes",
    "a photo of a coach wearing a jacket",
    "a photo of a person standing on the sideline in a coat",
]
COLOURS = ["red", "orange", "pink", "black", "white", "grey", "blue", "yellow", "green", "purple", "navy"]
_STATE: dict[str, Any] = {}


def _tensor(out):
    """transformers ≥ 5 returns a model output from get_*_features; older versions a tensor."""
    return out if hasattr(out, "norm") else out.pooler_output


def _load(device: str):
    if _STATE.get("device") == device:
        return _STATE["model"], _STATE["processor"], _STATE["direction"], _STATE["colours"], _STATE["scale"]
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel

    model = SiglipModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    with torch.no_grad():
        def mean_text(prompts):
            t = _tensor(model.get_text_features(**tok(prompts, padding="max_length", return_tensors="pt").to(device)))
            return torch.nn.functional.normalize(t, dim=-1).mean(0)

        direction = (mean_text(PLAYER_PROMPTS) - mean_text(OTHER_PROMPTS)).float().cpu().numpy()
        colours = torch.stack([mean_text([f"a photo of a person wearing a {c} shirt"]) for c in COLOURS])
        colours = colours.float().cpu().numpy()
        scale = float(model.logit_scale.exp().item())
    _STATE.update(device=device, model=model, processor=processor, direction=direction, colours=colours, scale=scale)
    return model, processor, direction, colours, scale


def crop_semantics(images_bgr: list[np.ndarray | None], device: str = "cpu") -> list[dict | None]:
    """Per crop: {"role_score": player − bystander, "colour_profile": [p per COLOURS]} (None when unavailable)."""
    idx = [i for i, im in enumerate(images_bgr) if im is not None and getattr(im, "size", 0)]
    out: list[dict | None] = [None] * len(images_bgr)
    if not idx:
        return out
    try:
        import torch

        from app.services.reid import torch_device

        dev = torch_device(device)
        model, processor, direction, colours, scale = _load(dev)
    except Exception as exc:  # noqa: BLE001 — optional model / offline
        logger.info("Crop semantics unavailable (%s) — keeping every person track, colour kit split", exc)
        return out
    for s in range(0, len(idx), BATCH):
        chunk = idx[s : s + BATCH]
        batch = [np.ascontiguousarray(images_bgr[i][:, :, ::-1]) for i in chunk]
        with torch.no_grad():
            emb = _tensor(model.get_image_features(**processor(images=batch, return_tensors="pt").to(dev)))
            emb = torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy()
        logits = scale * (emb @ colours.T)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        for i, e, p in zip(chunk, emb @ direction, probs):
            out[i] = {"role_score": float(e), "colour_profile": [round(float(v), 4) for v in p]}
    return out


def annotate_tracks(tracks: list, device: str = "cpu") -> int:
    """Fill features['role_score'] / ['colour_profile'] from each track's saved thumbnail.

    In memory only (the caller decides what to persist). Returns how many tracks carry them.
    """
    import cv2

    todo, imgs = [], []
    for t in tracks:
        feats = getattr(t, "features", None)
        if feats is None or feats.get("colour_profile") is not None:
            continue
        path = feats.get("thumb_path")
        img = cv2.imread(str(path)) if path and Path(str(path)).is_file() else None
        if img is not None:
            todo.append(t)
            imgs.append(img)
    for t, sem in zip(todo, crop_semantics(imgs, device)):
        if sem:
            t.features.update(sem)
    return sum(1 for t in tracks if (getattr(t, "features", None) or {}).get("colour_profile") is not None)
