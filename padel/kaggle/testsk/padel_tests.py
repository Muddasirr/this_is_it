"""CPU kernel: run the padel test suite against the current padel-code bundle (no models are retrained)."""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "pytest", "httpx", "ultralytics==8.4.189"],
               check=False)
code = Path(sorted(glob.glob("/kaggle/input/**/padel_analyzer/__init__.py", recursive=True))[0]).parents[1]
repo = Path("/tmp/padel")
shutil.copytree(code, repo)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=repo, env=dict(os.environ, PYTHONPATH=str(repo)),
                   capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
Path("/kaggle/working/pytest.log").write_text(r.stdout + r.stderr)
