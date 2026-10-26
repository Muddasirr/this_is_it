"""Group hits into rallies (points)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import Hit

MAX_GAP_S = 4.0  # cross-checked on PT100: rally count within 10 % on both finals (docs/EVAL.md §3)


@dataclass
class Rally:
    id: int
    start: int  # frame
    end: int
    hits: list[int] = field(default_factory=list)  # indices into the match hit list


def group(hits: list[Hit], fps: float, n_frames: int, segments: list[tuple[int, int]] | None = None,
          serves: list[bool] | None = None, max_gap_s: float = MAX_GAP_S) -> list[Rally]:
    """New rally when the gap between hits exceeds MAX_GAP_S, a scene cut separates them,
    or a serve follows at least 1.5 s after the previous hit."""
    seg_of = _segment_lookup(segments or [(0, n_frames)])
    rallies: list[Rally] = []
    for i, h in enumerate(hits):
        prev = hits[i - 1] if i else None
        new = (
            prev is None
            or (h.frame - prev.frame) / fps > max_gap_s
            or seg_of(h.frame) != seg_of(prev.frame)
            or (serves is not None and serves[i] and (h.frame - prev.frame) / fps > 1.5)
        )
        if new:
            rallies.append(Rally(len(rallies), h.frame, h.frame))
        rallies[-1].hits.append(i)
        rallies[-1].end = h.frame
    pad_pre, pad_post = int(1.0 * fps), int(1.5 * fps)
    for r in rallies:
        a, b = (segments or [(0, n_frames)])[seg_of(r.start)]
        r.start = max(a, r.start - pad_pre)
        r.end = min(b - 1, r.end + pad_post)
    return rallies


def _segment_lookup(segments: list[tuple[int, int]]):
    starts = [a for a, _ in segments]

    def seg_of(f: int) -> int:
        import bisect

        return max(0, bisect.bisect_right(starts, f) - 1)

    return seg_of
