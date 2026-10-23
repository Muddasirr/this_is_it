"""Jersey-number classifier (legible / tens / units heads on a ResNet-18 trunk).

Shared by training (app/scripts/train_jersey.py, SoccerNet GSR crops) and the
product reader so preprocessing can never drift. Weights: VISION_JERSEY_MODEL.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

CROP_H, CROP_W = 96, 64
MIN_BOX_H = 48


def upper_body(frame, xyxy) -> np.ndarray | None:
    """Shoulders-to-waist crop where shirt numbers sit, resized to CROP_H × CROP_W (BGR)."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    bh = y2 - y1
    if bh < MIN_BOX_H or x2 - x1 < 12:
        return None
    crop = frame[y1 + int(0.05 * bh) : y1 + int(0.6 * bh), x1:x2]
    return cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA) if crop.size else None


class JerseyNet:
    """Thin wrapper: torchvision ResNet-18 trunk + legible / tens / units heads."""

    @staticmethod
    def build(pretrained: bool = True):
        import torch.nn as nn
        import torchvision

        trunk = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        feat = trunk.fc.in_features
        trunk.fc = nn.Identity()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = trunk
                self.legible = nn.Linear(feat, 2)
                self.tens = nn.Linear(feat, 10)
                self.units = nn.Linear(feat, 10)

            def forward(self, x):
                f = self.trunk(x)
                return self.legible(f), self.tens(f), self.units(f)

        return Net()


def to_tensor(crops_bgr: np.ndarray):
    import torch

    x = torch.from_numpy(np.ascontiguousarray(crops_bgr[..., ::-1])).float().div(255.0).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


LEGIBLE_MIN = 0.6
DIGIT_MIN = 0.5
_NET: tuple[Any, str, str] | None = None


def _load(device: str):
    global _NET
    path = os.environ.get("VISION_JERSEY_MODEL", "")
    if not path or not os.path.isfile(path):
        return None
    if _NET is None or _NET[1] != device or _NET[2] != path:
        import torch

        ckpt = torch.load(path, map_location=device, weights_only=True)
        net = JerseyNet.build(pretrained=False)
        net.load_state_dict(ckpt["state_dict"])
        _NET = (net.to(device).eval(), device, path)
    return _NET[0]


def read_jersey_number_cls(frame, xyxy: list[float], device: str | None = None) -> dict[str, Any] | None:
    """{number, confidence} or None when the shirt number is not legible / model missing."""
    import torch

    from app.services.reid import torch_device

    dev = torch_device(device if device is not None else os.environ.get("VISION_DEVICE", "cpu"))
    net = _load(dev)
    crop = upper_body(frame, xyxy) if net is not None else None
    if crop is None:
        return None
    with torch.no_grad():
        lg, tn, un = (torch.softmax(o, dim=-1)[0] for o in net(to_tensor(crop[None]).to(dev)))
    p_leg = float(lg[1])
    t, u = int(tn.argmax()), int(un.argmax())
    conf = p_leg * float(tn[t]) * float(un[u])
    if p_leg < LEGIBLE_MIN or float(tn[t]) < DIGIT_MIN or float(un[u]) < DIGIT_MIN:
        return None
    number = t * 10 + u if t else u
    return {"number": number, "confidence": round(conf, 3)} if number > 0 else None
