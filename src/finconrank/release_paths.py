"""Release-only location adapter; no model, data or network access."""
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def release_path(variable: str, default: str, suffix: str = '') -> str:
    value = os.environ.get(variable, '').strip() or default
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    return str(root / suffix) if suffix else str(root)
