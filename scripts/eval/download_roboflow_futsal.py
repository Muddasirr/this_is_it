#!/usr/bin/env python3
"""Download Roboflow Futsal-Object-Detection_v2 (YOLOv8) → data/eval/roboflow_futsal.

Requires ROBOFLOW_API_KEY in env or .env.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "roboflow_futsal"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> None:
    _load_dotenv()
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("ROBOFLOW_API_KEY not set")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("pip install roboflow")
        sys.exit(1)

    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "helixs-project")
    project = os.environ.get("ROBOFLOW_PROJECT", "futsal-object-detection_v2")
    version = int(os.environ.get("ROBOFLOW_VERSION", "1"))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rf = Roboflow(api_key=key)
    ver = rf.workspace(workspace).project(project).version(version)
    print("downloading", workspace, project, version, "→", OUT)
    ds = ver.download("yolov8", location=str(OUT), overwrite=True)
    print("done", ds.location)
    yaml = Path(ds.location) / "data.yaml"
    assert yaml.exists(), f"missing {yaml}"
    print(yaml.read_text())


if __name__ == "__main__":
    main()
