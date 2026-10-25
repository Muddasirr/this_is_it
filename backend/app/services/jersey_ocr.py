"""Best-effort jersey number OCR from player crops.

Uses Tesseract when available; otherwise returns no number. Never crashes the pipeline.
Tesseract calls use a hard timeout — OCR was starving full-match GPU runs.

Critical: prefer full 2-digit numbers. Early exit on a lone "1"/"2" caused #13→1 and #23→2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
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


def jersey_reader():
    """The shirt-number reader the pipeline should use, or None.

    Trained classifier (app/services/jersey_cls.py) when VISION_JERSEY_MODEL points
    at weights. Tesseract only on explicit VISION_JERSEY_OCR=tesseract: on SoccerNet
    GSR valid (94 player tracks) it was right for 7 % of players and wrong 92 % of
    the times it answered — no consensus rule got it above 50 % — and a wrong number
    is worse than none (mislabels players, vetoes correct cross-camera links).
    """
    raw = os.environ.get("VISION_JERSEY_OCR", "1").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return None
    model = os.environ.get("VISION_JERSEY_MODEL", "")
    if model and os.path.isfile(model):
        from app.services.jersey_cls import read_jersey_number_cls

        return read_jersey_number_cls
    if raw == "tesseract" and _tesseract_ready():
        return read_jersey_number
    return None


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
    # Try variants until we have a confident 2-digit hit (avoid #13→1 early exit).
    variants = _prep_variants(crop, cv2)
    for img in variants:
        hits.extend(_ocr_full_string(img, pytesseract))
        if any(n >= 10 and c >= 70 for n, c in hits):
            break
    # Digit-box merge only if we still lack a 2-digit read
    if not any(n >= 10 for n, _ in hits):
        for img in variants[:1]:
            hits.extend(_ocr_merge_digit_boxes(img, pytesseract))

    best = _pick_best_number(hits)
    if best is None:
        return None
    num, conf = best
    return {"number": num, "confidence": round(min(0.95, conf / 100.0), 3)}


def _prep_variants(crop, cv2) -> list:
    """Otsu + inverted — white-on-red and dark-on-white numbers (keep fast)."""
    h, w = crop.shape[:2]
    # Prefer chest band — numbers live there; skips head noise
    y0, y1 = int(h * 0.15), int(h * 0.70)
    x0, x1 = int(w * 0.12), int(w * 0.88)
    roi = crop[y0:y1, x0:x1] if y1 > y0 + 12 and x1 > x0 + 12 else crop
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    return [otsu, inv]


def _parse_digit_string(compact: str) -> list[tuple[int, float]]:
    """Extract jersey candidates from a digit-only string. Prefer full 2-digit runs."""
    out: list[tuple[int, float]] = []
    if not compact:
        return out
    # Longest contiguous digit runs first (e.g. "132" → try 13, then 32)
    for m in re.finditer(r"[1-9]\d?", compact):
        token = m.group(0)
        try:
            num = int(token)
        except ValueError:
            continue
        if 10 <= num <= 99:
            out.append((num, 88.0))
        elif 1 <= num <= 9:
            out.append((num, 50.0))
    # Explicit 2-digit window at start (common for "23xx" noise)
    if len(compact) >= 2 and compact[:2].isdigit():
        num = int(compact[:2])
        if 10 <= num <= 99:
            out.append((num, 90.0))
        elif 1 <= num <= 9:
            out.append((num, 48.0))
    return out


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
        out.extend(_parse_digit_string(compact))
        for m in _NUMBER_RE.finditer(compact):
            num = int(m.group(1))
            if 1 <= num <= 99:
                out.append((num, 75.0 if num >= 10 else 45.0))
        # Stop PSM scan once we have a solid 2-digit candidate
        if any(n >= 10 and c >= 80 for n, c in out):
            break
    return out


def _ocr_merge_digit_boxes(img, pytesseract) -> list[tuple[int, float]]:
    """Merge left-to-right digit boxes so '1'+'3' → 13 and '2'+'3' → 23."""
    out: list[tuple[int, float]] = []
    try:
        data = pytesseract.image_to_data(
            img,
            config="--psm 6 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
            timeout=_OCR_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001
        return out

    digits: list[tuple[int, int, int, float]] = []  # (cx, x1, digit, conf)
    n = len(data.get("text") or [])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw or not raw.isdigit():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 40.0
        if conf < 20:
            continue
        x, w = int(data["left"][i]), int(data["width"][i])
        for ch in raw:
            if ch.isdigit():
                d = int(ch)
                if 0 <= d <= 9:
                    digits.append((x + w // 2, x, d, max(conf, 30.0)))
                    x += max(w // max(len(raw), 1), 1)

    if not digits:
        return out
    digits.sort(key=lambda t: t[0])

    # Emit singles
    for _, _, d, conf in digits:
        if 1 <= d <= 9:
            out.append((d, conf * 0.55))

    # Merge adjacent pairs into 10–99
    for i in range(len(digits) - 1):
        _, x1, d1, c1 = digits[i]
        cx2, x2, d2, c2 = digits[i + 1]
        gap = x2 - (x1 + abs(cx2 - x2))  # rough
        # Prefer small horizontal gap (same number)
        if digits[i + 1][0] - digits[i][0] > max(img.shape[1] * 0.45, 80):
            continue
        if d1 == 0:
            continue
        num = d1 * 10 + d2
        if 10 <= num <= 99:
            out.append((num, min(95.0, 0.55 * (c1 + c2) + 25.0)))
    return out


def _pick_best_number(hits: list[tuple[int, float]]) -> tuple[int, float] | None:
    if not hits:
        return None
    scores: dict[int, float] = {}
    for num, conf in hits:
        if num <= 0 or num > 99:
            continue
        # Strong bias toward 2-digit so #13/#23 win over tens digit alone
        bonus = 35.0 if num >= 10 else 0.0
        scores[num] = max(scores.get(num, 0.0), conf + bonus)

    # Drop tens digit when the full two-digit number is present
    for two in list(scores):
        if two < 10:
            continue
        tens = two // 10
        ones = two % 10
        if tens in scores and scores[two] >= scores[tens] * 0.55:
            scores.pop(tens, None)
        if ones in scores and ones != 0 and scores[two] >= scores[ones] * 0.55:
            scores.pop(ones, None)

    if not scores:
        return None
    best_num = max(scores.keys(), key=lambda n: (scores[n], 1 if n >= 10 else 0, n))
    raw = scores[best_num] - (35.0 if best_num >= 10 else 0.0)
    return best_num, max(raw, 40.0)


def prefer_jersey_number(votes: list[tuple[int, float]]) -> tuple[int | None, float]:
    """Majority vote that does not let '1' beat '13' or '2' beat '23'."""
    if not votes:
        return None, 0.0
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for num, conf in votes:
        if num <= 0 or num > 99:
            continue
        weight = float(conf) * (1.6 if num >= 10 else 1.0)
        scores[num] = scores.get(num, 0.0) + weight
        counts[num] = counts.get(num, 0) + 1

    for two in list(scores):
        if two < 10:
            continue
        tens = two // 10
        ones = two % 10
        if tens in scores and counts.get(two, 0) >= 1:
            if scores[two] >= scores[tens] * 0.45:
                scores.pop(tens, None)
        if ones in scores and ones != 0 and counts.get(two, 0) >= 1:
            if scores[two] >= scores[ones] * 0.45:
                scores.pop(ones, None)

    if not scores:
        return None, 0.0
    best = max(scores.keys(), key=lambda n: (scores[n], counts.get(n, 0), 1 if n >= 10 else 0))
    avg = scores[best] / max(1, counts.get(best, 1))
    if best >= 10:
        avg = avg / 1.6
    return best, round(min(0.95, avg), 3)


# --- TrOCR reader (transformer scene-text OCR) ------------------------------------
TROCR_MODEL = os.environ.get("VISION_TROCR_MODEL", "microsoft/trocr-small-printed")
TROCR_MIN_CONF = 0.5
_TROCR: tuple[Any, Any, str] | None = None


def _trocr(device: str = "cpu"):
    global _TROCR
    if _TROCR is None or _TROCR[2] != device:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device).eval()
        _TROCR = (processor, model, device)
    return _TROCR


def read_jersey_number_trocr(frame, xyxy: list[float], device: str | None = None) -> dict[str, Any] | None:
    """{number, confidence} from the torso of a person box via TrOCR; None if illegible.

    Confidence = mean token probability of the decoded string; low-confidence or
    non-numeric reads are dropped (the model's own legibility filter).
    """
    import cv2
    import numpy as np
    import torch

    from app.services.reid import torch_device

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 36:
        return None
    crop = frame[y1 + int(bh * 0.12) : y1 + int(bh * 0.58), x1 + int(bw * 0.12) : x1 + int(bw * 0.88)]
    if crop.size == 0:
        return None
    if crop.shape[0] < 64:
        s = 64.0 / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 64), interpolation=cv2.INTER_CUBIC)
    dev = torch_device(device if device is not None else os.environ.get("VISION_DEVICE", "cpu"))
    try:
        processor, model, _ = _trocr(dev)
    except Exception as exc:  # noqa: BLE001 — optional model
        logger.info("TrOCR unavailable (%s)", exc)
        return None
    pixel_values = processor(images=np.ascontiguousarray(crop[:, :, ::-1]), return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        out = model.generate(pixel_values, max_new_tokens=4, output_scores=True, return_dict_in_generate=True)
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    m = _NUMBER_RE.fullmatch(text.replace(" ", ""))
    if not m:
        return None
    probs = [float(torch.softmax(s, dim=-1).max()) for s in out.scores]
    conf = float(np.mean(probs)) if probs else 0.0
    if conf < TROCR_MIN_CONF:
        return None
    return {"number": int(m.group(1)), "confidence": round(conf, 3)}


# PARSeq (vendored, Apache-2.0 — app/vendor/parseq): commercially usable scene-text
# recogniser; replaces the CC BY-NC jersey-number pipeline we cannot ship.
PARSEQ_WEIGHTS = os.environ.get("VISION_PARSEQ_WEIGHTS", str(Path(__file__).resolve().parents[2] / "models" / "parseq-bb5792a6.pt"))
PARSEQ_MIN_CONF = 0.6
_PARSEQ: tuple[Any, Any, str] | None = None


def _parseq(device: str = "cpu"):
    global _PARSEQ
    if _PARSEQ is None or _PARSEQ[2] != device:
        from app.vendor import parseq

        model, tok = parseq.load(PARSEQ_WEIGHTS, device)
        _PARSEQ = (model, tok, device)
    return _PARSEQ


def read_jersey_number_parseq(frame, xyxy: list[float], device: str | None = None) -> dict[str, Any] | None:
    """{number, confidence} from the torso of a person box via PARSeq; None if illegible.

    Confidence = product of the per-character probabilities; anything that is not a
    clean 1–2 digit number at PARSEQ_MIN_CONF is dropped (a wrong number is worse than none).
    """
    import torch

    from app.services.reid import torch_device
    from app.vendor import parseq

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 16 or bh < 36:
        return None
    crop = frame[y1 + int(bh * 0.12) : y1 + int(bh * 0.58), x1 + int(bw * 0.12) : x1 + int(bw * 0.88)]
    if crop.size == 0:
        return None
    dev = torch_device(device if device is not None else os.environ.get("VISION_DEVICE", "cpu"))
    try:
        model, tok, _ = _parseq(dev)
    except Exception as exc:  # noqa: BLE001 — weights missing / checksum mismatch
        logger.info("PARSeq unavailable (%s)", exc)
        return None
    with torch.no_grad():
        probs = model(tok, parseq.preprocess([crop]).to(dev)).softmax(-1)
    labels, confs = tok.decode(probs)
    text, char_p = labels[0].replace(" ", ""), confs[0]
    m = _NUMBER_RE.fullmatch(text)
    if not m:
        return None
    conf = float(char_p.prod()) if len(char_p) else 0.0
    if conf < PARSEQ_MIN_CONF:
        return None
    return {"number": int(m.group(1)), "confidence": round(conf, 3)}
