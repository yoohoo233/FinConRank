
# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))

from finconrank.release_paths import release_path
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(release_path('FINCONRANK_WORKSPACE', '.', ''))
RUN = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".atomic_")
    with os.fdopen(fd, "w") as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def csvfile(path, data, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(dict.fromkeys(k for row in data for k in row)) or ["status"]
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".atomic_")
    with os.fdopen(fd, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_protocol():
    path = RUN / "01_PROTOCOL_AND_INPUTS/protocol_lock.json"
    expected = (path.parent / "protocol_lock.sha256").read_text().split()[0]
    if sha(path) != expected:
        raise RuntimeError("protocol changed")
    protocol = read(path)
    for source in protocol["script_hashes"]:
        if sha(source["path"]) != source["sha256"]:
            raise RuntimeError(f"script changed: {source['path']}")
    return protocol
