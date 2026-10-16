"""Redirect the detached scheduler to its durable log, then exec it."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import os
import sys
from pathlib import Path

run = Path(__file__).resolve().parents[2]
log = run / "06_REPRODUCIBILITY/logs/scheduler.log"
log.parent.mkdir(parents=True, exist_ok=True)
handle = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(handle, 1)
os.dup2(handle, 2)
os.close(handle)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name("scheduler_v2.py"))])
