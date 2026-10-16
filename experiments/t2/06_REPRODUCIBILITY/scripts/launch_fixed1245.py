"""Detached fixed1245 dispatcher, preserving live workers and locked protocol."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import os
import sys
from pathlib import Path
from t2_common import sha

folder = Path(__file__).resolve().parent
scheduler = folder / "scheduler_fixed1245.py"
if sha(scheduler) != "5a9497db0940eaeffd495df510f73f9249c9e3977b6eeaed9474117aa5546f22":
    raise RuntimeError("GPU affinity dispatcher hash mismatch")
log = folder.parents[1] / "06_REPRODUCIBILITY/logs/scheduler_fixed1245.log"
handle = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(handle, 1)
os.dup2(handle, 2)
os.close(handle)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.execv(sys.executable, [sys.executable, str(scheduler)])
