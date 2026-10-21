#!/usr/bin/env python3
"""Phase 2 gate: kit A/B lightness split (unit + optional live match).

  cd backend && python -m pytest tests/test_kit_identity.py -q
  python scripts/eval/check_kit_split.py --match-id <uuid>   # via docker api
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_kit_identity.py", "-q"],
        cwd=ROOT / "backend",
    )
    if r.returncode != 0:
        raise SystemExit("PHASE2_KIT_GATE FAIL (unit tests)")
    print("PHASE2_KIT_GATE PASS (unit: white vs dark/red not same kit)")


if __name__ == "__main__":
    main()
