"""Best-effort jersey number OCR from player crops (CPU-friendly).

Uses Tesseract when available; otherwise returns no number. Never crashes the pipeline.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\b([0-9]{1,2})\b")


@lru_cache(maxsize=1)
def _tesseract_ready() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        logger.info("Tesseract/pytesseract not available — jersey OCR disabled")
        return False


def read_jersey_number(frame, xyxy: list[float]) -> dict[str, Any] | None:
    """Return {number, confidence} or None."""
    if not _tesseract_ready():
        return None
    import cv2
    import numpy as np
    import pytesseract

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 28 or bh < 40:
        return None

    # torso / shirt region
    y_a = y1 + int(bh * 0.18)
    y_b = y1 + int(bh * 0.62)
    x_a = x1 + int(bw * 0.18)
    x_b = x2 - int(bw * 0.18)
    if y_b <= y_a + 8 or x_b <= x_a + 8:
        return None
    crop = frame[y_a:y_b, x_a:x_b]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # also try inverted (dark numbers on light kits / light on dark)
    candidates = [thr, cv2.bitwise_not(thr)]

    best_num: int | None = None
    best_conf = 0.0
    for img in candidates:
        try:
            data = pytesseract.image_to_data(
                img,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:  # noqa: BLE001
            continue
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                continue
            if c < 40:
                continue
            m = _NUMBER_RE.search(text or "")
            if not m:
                continue
            num = int(m.group(1))
            if num <= 0 or num > 99:
                continue
            if c > best_conf:
                best_conf = c
                best_num = num

    if best_num is None:
        return None
    return {"number": best_num, "confidence": round(min(0.95, best_conf / 100.0), 3)}
