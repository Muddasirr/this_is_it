"""Vendored PARSeq: refuses unpinned weights; reads a clear number when the pinned weights are present."""
from pathlib import Path

import pytest

from app.vendor import parseq

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "parseq-bb5792a6.pt"


def test_rejects_weights_with_wrong_checksum(tmp_path):
    bad = tmp_path / "parseq.pt"
    bad.write_bytes(b"not the release")
    with pytest.raises(ValueError, match="SHA-256"):
        parseq.load(bad)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="pinned PARSeq weights not present")
def test_reads_a_printed_shirt_number():
    import cv2
    import numpy as np

    from app.services.jersey_ocr import read_jersey_number_parseq

    frame = np.full((300, 200, 3), 200, np.uint8)
    cv2.rectangle(frame, (40, 40), (160, 280), (30, 30, 160), -1)  # a red shirt, box = whole person
    cv2.putText(frame, "23", (62, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 5)
    out = read_jersey_number_parseq(frame, [40, 40, 160, 280], device="cpu")
    assert out and out["number"] == 23
