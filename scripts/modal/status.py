"""Status / results of spawned Modal calls recorded in data/modal_stage/calls.json."""
import json
import sys
from pathlib import Path

import modal

calls = json.loads(Path("data/modal_stage/calls.json").read_text())
for name, cid in calls.items():
    fc = modal.FunctionCall.from_id(cid)
    try:
        out = fc.get(timeout=0)
        print(name, "DONE", json.dumps(out)[: int(sys.argv[1]) if len(sys.argv) > 1 else 1500])
    except TimeoutError:
        print(name, "running")
    except Exception as exc:  # noqa: BLE001
        print(name, "FAILED", str(exc)[:500])
