"""Futsal court format → expected roster / on-court counts (5v5, 6v6, 7v7)."""

from __future__ import annotations

import os

_FORMATS = {
    "5v5": {"per_side": 5, "max_auto_players": 16, "max_on_court": 14},
    "6v6": {"per_side": 6, "max_auto_players": 18, "max_on_court": 16},
    "7v7": {"per_side": 7, "max_auto_players": 20, "max_on_court": 18},
}


def get_court_format() -> str:
    raw = os.environ.get("COURT_FORMAT", "").strip().lower().replace(" ", "")
    if not raw:
        try:
            from app.core.config import get_settings

            raw = (get_settings().court_format or "5v5").strip().lower().replace(" ", "")
        except Exception:
            raw = "5v5"
    return raw if raw in _FORMATS else "5v5"


def court_settings(fmt: str | None = None) -> dict[str, int | str]:
    key = fmt or get_court_format()
    cfg = dict(_FORMATS.get(key, _FORMATS["5v5"]))
    cfg["format"] = key if key in _FORMATS else "5v5"
    return cfg


def max_auto_players() -> int:
    return int(court_settings()["max_auto_players"])


def max_on_court() -> int:
    return int(court_settings()["max_on_court"])
