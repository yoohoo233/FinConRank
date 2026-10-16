#!/usr/bin/env python3
"""Persist GPU snapshots while a known cost process is alive."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from major_revision_gpu_scheduler import parse_inventory, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("a") as handle:
        while True:
            alive = Path(f"/proc/{args.pid}/cmdline").exists()
            command = Path(f"/proc/{args.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace") if alive else ""
            if alive and "measure_major_revision_serial_cost.py" not in command:
                raise RuntimeError("PID identity changed")
            handle.write(json.dumps({"at": utc_now(), "pid": args.pid, "alive": alive,
                                     "gpus": [asdict(gpu) for gpu in parse_inventory()]}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not alive:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
