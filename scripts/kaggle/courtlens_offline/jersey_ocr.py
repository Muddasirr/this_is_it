"""Best-effort jersey number OCR from player crops.

Uses Tesseract when available; otherwise returns no number. Never crashes the pipeline.
Tesseract calls use a hard timeout — OCR was starving full-match GPU runs.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(?<!\d)([1-9]\d?)(?!\d)")
_OCR_TIMEOUT_S = float(os.environ.get("VISION_OCR_TIMEOUT_S", "1.0"))


@lru_cache(maxsize=1)
def _tesseract_ready() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        logger.info("Tesseract/pytesseract not available — jersey OCR disabled")
        return False


def jersey_ocr_enabled() -> bool:
    """VISION_JERSEY_OCR=0|false|off disables all OCR."""
    raw = os.environ.get("VISION_JERSEY_OCR", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"} and _tesseract_ready()


def read_jersey_number(frame, xyxy: list[float]) -> dict[str, Any] | None:
    """Return {number, confidence} or None from a person bbox on a video frame."""
    if not jersey_ocr_enabled():
        return None
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 28 or bh < 40:
        return None

    fx1, fy1, fx2, fy2 = 0.12, 0.12, 0.88, 0.58
    x_a = x1 + int(bw * fx1)
    y_a = y1 + int(bh * fy1)
    x_b = x1 + int(bw * fx2)
    y_b = y1 + int(bh * fy2)
    if y_b <= y_a + 8 or x_b <= x_a + 8:
        return None
    crop = frame[y_a:y_b, x_a:x_b]
    if crop.size == 0:
        return None
    return read_jersey_number_bgr(crop)


def read_jersey_number_bgr(crop) -> dict[str, Any] | None:
    """OCR a BGR crop (player thumb or torso). Used for gallery post-pass too."""
    if not jersey_ocr_enabled():
        return None
    import cv2
    import pytesseract

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h, w = crop.shape[:2]
    if w < 24 or h < 24:
        return None
    hits: list[tuple[int, float]] = []
    for img in _prep_variants(crop, cv2):
        hits.extend(_ocr_full_string(img, pytesseract))
        if hits:
            break
    best = _pick_best_number(hits)
    if best is None:
        return None
    num, conf = best
    return {"number": num, "confidence": round(min(0.95, conf / 100.0), 3)}


def _prep_variants(crop, cv2) -> list:
    """Otsu + inverted + adaptive — white-on-red and dark-on-white numbers."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return [otsu, inv, adapt]


def _ocr_full_string(img, pytesseract) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for psm in ("7", "8", "13"):
        try:
            text = pytesseract.image_to_string(
                img,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
                timeout=_OCR_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001
            continue
        compact = re.sub(r"\D", "", text or "")
        if not compact:
            continue
        if len(compact) >= 2:
            try:
                num = int(compact[:2])
            except ValueError:
                continue
            if 10 <= num <= 99:
                out.append((num, 82.0))
            elif 1 <= num <= 9:
                out.append((num, 52.0))
        elif compact.isdigit():
            num = int(compact)
            if 1 <= num <= 99:
                out.append((num, 55.0 if num < 10 else 72.0))
        for m in _NUMBER_RE.finditer(compact):
            num = int(m.group(1))
            if 1 <= num <= 99:
                out.append((num, 72.0 if num >= 10 else 48.0))
        if out:
            break
    return out


def _pick_best_number(hits: list[tuple[int, float]]) -> tuple[int, float] | None:
    if not hits:
        return None
    scores: dict[int, float] = {}
    for num, conf in hits:
        if num <= 0 or num > 99:
            continue
        bonus = 20.0 if num >= 10 else 0.0
        scores[num] = max(scores.get(num, 0.0), conf + bonus)

    for two in list(scores):
        if two < 10:
            continue
        tens = two // 10
        if tens in scores and scores[two] >= scores[tens] * 0.7:
            scores.pop(tens, None)

    if not scores:
        return None
    best_num = max(scores.keys(), key=lambda n: (scores[n], 1 if n >= 10 else 0, n))
    raw = scores[best_num] - (20.0 if best_num >= 10 else 0.0)
    return best_num, max(raw, 40.0)


def prefer_jersey_number(votes: list[tuple[int, float]]) -> tuple[int | None, float]:
    """Majority vote that does not let '1' beat '10'."""
    if not votes:
        return None, 0.0
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for num, conf in votes:
        if num <= 0 or num > 99:
            continue
        weight = float(conf) * (1.4 if num >= 10 else 1.0)
        scores[num] = scores.get(num, 0.0) + weight
        counts[num] = counts.get(num, 0) + 1

    for two in list(scores):
        if two < 10:
            continue
        tens = two // 10
        if tens in scores and counts.get(two, 0) >= 1:
            if scores[two] >= scores[tens] * 0.5:
                scores.pop(tens, None)

    if not scores:
        return None, 0.0
    best = max(scores.keys(), key=lambda n: (scores[n], counts.get(n, 0), 1 if n >= 10 else 0))
    avg = scores[best] / max(1, counts.get(best, 1))
    if best >= 10:
        avg = avg / 1.4
    return best, round(min(0.95, avg), 3)
