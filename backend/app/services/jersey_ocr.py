"""Best-effort jersey number OCR from player crops (CPU-friendly).

Uses Tesseract when available; otherwise returns no number. Never crashes the pipeline.
Common CCTV failure mode: reading "10" as "1" — we join adjacent digits and prefer 2-digit numbers.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(?<!\d)([1-9]\d?)(?!\d)")


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
    import pytesseract

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 28 or bh < 40:
        return None

    # Two ROIs only — more variants made CPU OCR too slow for full-match runs
    rois = [
        (0.12, 0.12, 0.88, 0.58),
        (0.10, 0.18, 0.90, 0.70),
    ]
    hits: list[tuple[int, float]] = []
    for fx1, fy1, fx2, fy2 in rois:
        x_a = x1 + int(bw * fx1)
        y_a = y1 + int(bh * fy1)
        x_b = x1 + int(bw * fx2)
        y_b = y1 + int(bh * fy2)
        if y_b <= y_a + 8 or x_b <= x_a + 8:
            continue
        crop = frame[y_a:y_b, x_a:x_b]
        if crop.size == 0:
            continue
        for img in _prep_variants(crop, cv2):
            hits.extend(_ocr_joined_digits(img, pytesseract))
            hits.extend(_ocr_full_string(img, pytesseract))
        if _pick_best_number(hits) and any(n >= 10 for n, _ in hits):
            break  # early exit once a 2-digit read lands

    best = _pick_best_number(hits)
    if best is None:
        return None
    num, conf = best
    return {"number": num, "confidence": round(min(0.95, conf / 100.0), 3)}


def _prep_variants(crop, cv2) -> list:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu + invert only — adaptive/morph stacks were too slow for every player box
    return [otsu, cv2.bitwise_not(otsu)]


def _ocr_full_string(img, pytesseract) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for psm in ("7", "8"):
        try:
            text = pytesseract.image_to_string(
                img,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
            )
        except Exception:  # noqa: BLE001
            continue
        compact = re.sub(r"\D", "", text or "")
        if not compact:
            continue
        # Whole string as number (handles "10" kept intact)
        if len(compact) >= 2:
            try:
                num = int(compact[:2])
            except ValueError:
                continue
            if 10 <= num <= 99:
                out.append((num, 78.0))
            elif 1 <= num <= 9:
                out.append((num, 52.0))
        elif compact.isdigit():
            num = int(compact)
            if 1 <= num <= 99:
                out.append((num, 55.0 if num < 10 else 70.0))
        for m in _NUMBER_RE.finditer(compact):
            num = int(m.group(1))
            if 1 <= num <= 99:
                out.append((num, 70.0 if num >= 10 else 48.0))
    return out


def _ocr_joined_digits(img, pytesseract) -> list[tuple[int, float]]:
    """Join spatially adjacent digit boxes so '1'+'0' → 10, not orphan '1'."""
    out: list[tuple[int, float]] = []
    try:
        data = pytesseract.image_to_data(
            img,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:  # noqa: BLE001
        return out

    tokens: list[tuple[int, int, str, float]] = []
    for text, conf, left, width in zip(
        data.get("text", []),
        data.get("conf", []),
        data.get("left", []),
        data.get("width", []),
    ):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        t = (text or "").strip()
        if c < 25 or not t.isdigit():
            continue
        tokens.append((int(left), int(left) + max(1, int(width)), t, c))

    for left, right, t, c in tokens:
        if len(t) >= 2:
            try:
                num = int(t[:2])
            except ValueError:
                continue
            if 1 <= num <= 99:
                out.append((num, c + (10.0 if num >= 10 else 0.0)))

    singles = [(l, r, int(t), c) for l, r, t, c in tokens if len(t) == 1]
    singles.sort(key=lambda x: x[0])
    i = 0
    while i < len(singles):
        l1, r1, d1, c1 = singles[i]
        if d1 == 0:
            i += 1
            continue
        if i + 1 < len(singles):
            l2, r2, d2, c2 = singles[i + 1]
            gap = l2 - r1
            char_w = max(1.0, float(r1 - l1))
            if 0 <= gap <= char_w * 2.0:
                num = d1 * 10 + d2
                if 10 <= num <= 99:
                    out.append((num, (c1 + c2) / 2.0 + 12.0))
                    i += 2
                    continue
        out.append((d1, c1))
        i += 1
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

    # Drop lone digit if a 2-digit number sharing that tens digit scored well
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
