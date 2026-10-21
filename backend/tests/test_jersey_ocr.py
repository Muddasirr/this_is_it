"""Phase 2: jersey OCR must keep full 2-digit numbers (#13 not 1, #23 not 2)."""

from __future__ import annotations


def test_pick_best_prefers_13_over_1():
    from app.services.jersey_ocr import _pick_best_number

    num, conf = _pick_best_number([(1, 80.0), (13, 70.0), (3, 40.0)])
    assert num == 13, f"got {num}"
    assert conf >= 40


def test_pick_best_prefers_23_over_2():
    from app.services.jersey_ocr import _pick_best_number

    num, _ = _pick_best_number([(2, 85.0), (23, 72.0)])
    assert num == 23


def test_prefer_vote_13_beats_1():
    from app.services.jersey_ocr import prefer_jersey_number

    num, _ = prefer_jersey_number([(1, 0.8), (1, 0.7), (13, 0.65), (13, 0.6)])
    assert num == 13


def test_prefer_vote_23_beats_2():
    from app.services.jersey_ocr import prefer_jersey_number

    num, _ = prefer_jersey_number([(2, 0.9), (23, 0.7), (2, 0.5)])
    assert num == 23


def test_parse_digit_string_keeps_two_digits():
    from app.services.jersey_ocr import _parse_digit_string

    hits = _parse_digit_string("13")
    assert any(n == 13 for n, _ in hits)
    hits23 = _parse_digit_string("23")
    assert any(n == 23 for n, _ in hits23)


def test_synthetic_digits_ocr_when_tesseract_available():
    """Draw '13' and '23' — must not truncate to 1 / 2."""
    import os

    os.environ.pop("VISION_JERSEY_OCR", None)
    from app.services.jersey_ocr import jersey_ocr_enabled, read_jersey_number_bgr

    if not jersey_ocr_enabled():
        return

    import cv2
    import numpy as np

    def _draw(text: str):
        img = np.full((160, 120, 3), (40, 40, 200), dtype=np.uint8)  # red-ish BGR
        cv2.putText(img, text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 5)
        return img

    for expected in (13, 23, 10, 7):
        hit = read_jersey_number_bgr(_draw(str(expected)))
        assert hit is not None, f"no OCR for {expected}"
        assert int(hit["number"]) == expected, f"expected {expected} got {hit}"
