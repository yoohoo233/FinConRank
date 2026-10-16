"""Detached operational dispatcher with the user-requested GPU affinity."""

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
scheduler = folder / "scheduler_fixed1234.py"
expected = "0af99ca2bfbe5d52a7540e3d331a2c19f02ae1762295e0384b0c274915ea5789"
if sha(scheduler) != expected:
    raise RuntimeError("GPU affinity dispatcher hash mismatch")
log = folder.parents[1] / "06_REPRODUCIBILITY/logs/scheduler_fixed1234.log"
log.parent.mkdir(parents=True, exist_ok=True)
handle = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.dup2(handle, 1)
os.dup2(handle, 2)
os.close(handle)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.execv(sys.executable, [sys.executable, str(scheduler)])
