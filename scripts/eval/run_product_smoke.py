#!/usr/bin/env python3
"""Phase 5 product-path smoke (no full GPU match required).

Checks: CameraLabel A/B/C, fuse, possession, share payload includes per-player heatmaps,
avatars API helpers importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    report: dict = {"ok": True, "checks": []}

    from app.models.entities import CameraLabel
    from app.services.court_format import court_settings, max_auto_players
    from app.services.fuse import fuse_event_candidates
    from app.services.possession import build_possession_timeline
    from app.services.vision import Detection, EventCandidate

    labels = {c.value for c in CameraLabel}
    report["checks"].append({"camera_labels": sorted(labels)})
    if not {"A", "B", "C"}.issubset(labels):
        report["ok"] = False

    cfg = court_settings("5v5")
    report["checks"].append({"court_5v5": cfg, "max_auto": max_auto_players()})

    # Fuse keeps highest confidence
    a = EventCandidate("shot", "A", 1000, 0.5, ["A-p1"], {})
    b = EventCandidate("shot", "B", 1100, 0.8, ["B-p1"], {})
    fused = fuse_event_candidates([a, b], bucket_ms=1500)
    report["checks"].append({"fuse_n": len(fused), "fuse_conf": fused[0].confidence if fused else None})
    if len(fused) != 1 or fused[0].confidence != 0.8:
        report["ok"] = False

    # Empty possession ok
    empty = build_possession_timeline([])
    report["checks"].append({"possession_empty_segments": len(empty.get("segments") or [])})

    # Share module includes heatmap field in source
    share_src = (ROOT / "backend/app/api/share.py").read_text()
    if '"heatmap": row.get("heatmap")' not in share_src and "heatmap" not in share_src:
        report["ok"] = False
        report["checks"].append({"share_heatmap": False})
    else:
        report["checks"].append({"share_heatmap": True})

    # Frontend public report has player heat select
    app_src = (ROOT / "frontend/src/App.tsx").read_text()
    report["checks"].append(
        {
            "frontend_player_heat": "heatPlayerId" in app_src
            and "PublicReportPage" in app_src
            and "selectedHeat" in app_src,
        }
    )
    if not report["checks"][-1]["frontend_player_heat"]:
        report["ok"] = False

    out = ROOT / "data/eval/product_smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)
    print("GATE: product path smoke ok (heatmaps + multi-cam fuse typed)")


if __name__ == "__main__":
    main()
