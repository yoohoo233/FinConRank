#!/usr/bin/env python3
"""Run fixed DocReRank GPU shards with conservative physical-GPU leases."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from major_revision_gpu_scheduler import (
    MAX_PHYSICAL_GPUS,
    GPUStatus,
    acquire,
    atomic_json,
    parse_inventory,
    read_leases,
    release,
    select_stable_gpus,
)


PYTHON = _release_sys.executable
FIXED_WORKERS = 4
REQUIRED_FREE_MIB = 18 * 1024
OBSERVATIONS = 3
OBSERVATION_SECONDS = 10
POLL_SECONDS = 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def event(path: Path, kind: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": now(), "event": kind, **fields}, ensure_ascii=False) + "\n")


def snapshot(output_dir: Path, observation: str, reserved: set[str]) -> list[GPUStatus]:
    statuses = [replace(item, scheduler_reserved=item.uuid in reserved) for item in parse_inventory()]
    rows = []
    for item in statuses:
        row = asdict(item)
        row.update({"captured_at": now(), "observation": observation})
        rows.append(row)
    append_csv(output_dir / "gpu_monitor.csv", rows)
    return statuses


def stable_snapshot(output_dir: Path, reserved: set[str], label: str) -> list[GPUStatus]:
    observations = []
    for index in range(OBSERVATIONS):
        observations.append(snapshot(output_dir, f"{label}_{index + 1}", reserved))
        if index + 1 < OBSERVATIONS:
            time.sleep(OBSERVATION_SECONDS)
    return select_stable_gpus(observations, set(), REQUIRED_FREE_MIB, requested=MAX_PHYSICAL_GPUS)


def write_inventory(path: Path, statuses: list[GPUStatus]) -> None:
    rows = [asdict(item) for item in statuses]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_complete(stats_path: Path, output_path: Path) -> bool:
    if not stats_path.is_file() or not output_path.is_file():
        return False
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return stats.get("status") == "complete" and stats.get("output_sha256") == sha256(output_path)
    except (OSError, ValueError, KeyError):
        return False


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_command(revision_run: Path, worker: int, output: Path, stats: Path, pilot: bool) -> list[str]:
    command = [
        PYTHON, "scripts/run_major_revision_docrerank.py", "worker",
        "--revision-run", str(revision_run),
        "--worker-index", str(worker),
        "--num-workers", "1" if pilot else str(FIXED_WORKERS),
        "--output", str(output), "--stats", str(stats),
    ]
    if pilot:
        command.extend(["--max-queries", "2"])
    return command


def run_queue(revision_run: Path, mode: str) -> None:
    scheduling = revision_run / "gpu_scheduling"
    result_dir = revision_run / "results" / "docrerank"
    worker_dir = result_dir / ("pilot" if mode == "pilot" else "workers")
    log_dir = revision_run / "logs" / "docrerank_workers"
    for path in (scheduling, worker_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    lease_path = scheduling / "gpu_leases.json"
    events_path = scheduling / "scheduler_events.jsonl"
    manifest_path = scheduling / "run_manifest.csv"
    existing = read_leases(lease_path)
    active = [row for row in existing.get("leases", []) if row.get("status") == "active" and Path(f"/proc/{row['pid']}").exists()]
    if active:
        raise RuntimeError(f"active major-revision GPU leases already exist: {active}")
    initial = parse_inventory()
    write_inventory(scheduling / "gpu_inventory.csv", initial)
    pending = [0] if mode == "pilot" else list(range(FIXED_WORKERS))
    if mode == "full":
        pending = [worker for worker in pending if not valid_complete(
            worker_dir / f"worker_{worker}.stats.json", worker_dir / f"worker_{worker}.jsonl"
        )]
    running: dict[int, dict[str, Any]] = {}
    attempts = {worker: 0 for worker in pending}
    max_concurrent = 0
    event(events_path, "scheduler_started", mode=mode, pending=pending, pid=os.getpid())
    while pending or running:
        reserved = {job["gpu"].uuid for job in running.values()}
        available_ram_gib = psutil.virtual_memory().available / (1024 ** 3)
        capacity = min(MAX_PHYSICAL_GPUS - len(running), max(0, int((available_ram_gib - 16) // 24)))
        if pending and capacity > 0:
            eligible = stable_snapshot(scheduling, reserved, f"claim_{mode}")
            eligible = [gpu for gpu in eligible if gpu.uuid not in reserved][:capacity]
            for gpu in eligible:
                if not pending:
                    break
                worker = pending.pop(0)
                attempts[worker] += 1
                output = worker_dir / ("pilot_predictions.jsonl" if mode == "pilot" else f"worker_{worker}.jsonl")
                stats = worker_dir / ("pilot_stats.json" if mode == "pilot" else f"worker_{worker}.stats.json")
                log_path = log_dir / ("pilot.log" if mode == "pilot" else f"worker_{worker}.log")
                command = worker_command(revision_run, worker, output, stats, mode == "pilot")
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu.index),
                    "FINCONRANK_PHYSICAL_GPU_ID": str(gpu.index),
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONPATH": "src:scripts:.",
                })
                log_handle = log_path.open("a", encoding="utf-8")
                started = now()
                process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                task = f"docrerank_{mode}_worker_{worker}_attempt_{attempts[worker]}"
                acquire(lease_path, revision_run.name, task, process.pid, gpu)
                running[worker] = {
                    "process": process, "gpu": gpu, "task": task, "command": command,
                    "output": output, "stats": stats, "log": log_path, "log_handle": log_handle,
                    "started_at": started,
                }
                event(events_path, "worker_started", mode=mode, worker=worker, pid=process.pid, gpu=asdict(gpu), command=command)
                max_concurrent = max(max_concurrent, len(running))
        if not running and pending:
            event(events_path, "waiting_for_gpu_or_ram", pending=pending, available_ram_gib=available_ram_gib)
            time.sleep(POLL_SECONDS)
            continue
        if running:
            time.sleep(POLL_SECONDS if mode == "full" else 5)
            snapshot(scheduling, f"poll_{mode}", {job["gpu"].uuid for job in running.values()})
            finished = [worker for worker, job in running.items() if job["process"].poll() is not None]
            for worker in finished:
                job = running.pop(worker)
                rc = job["process"].returncode
                job["log_handle"].close()
                success = rc == 0 and valid_complete(job["stats"], job["output"])
                release(lease_path, job["task"], "complete" if success else "failed")
                row = {
                    "mode": mode, "worker": worker, "attempt": attempts[worker],
                    "physical_gpu_id": job["gpu"].index, "gpu_uuid": job["gpu"].uuid,
                    "pid": job["process"].pid, "started_at": job["started_at"], "finished_at": now(),
                    "exit_code": rc, "status": "complete" if success else "failed",
                    "output": str(job["output"]), "stats": str(job["stats"]), "log": str(job["log"]),
                    "command": " ".join(job["command"]),
                }
                append_csv(manifest_path, [row])
                event(events_path, "worker_finished", **row)
                if not success:
                    log_text = job["log"].read_text(encoding="utf-8", errors="replace")[-8000:]
                    if "out of memory" in log_text.lower():
                        raise RuntimeError(f"worker {worker} OOM; scientific batch setting left unchanged")
                    if attempts[worker] < 3:
                        pending.append(worker)
                    else:
                        raise RuntimeError(f"worker {worker} failed after three attempts")
    summary = {
        "mode": mode, "finished_at": now(), "status": "complete",
        "fixed_worker_count": 1 if mode == "pilot" else FIXED_WORKERS,
        "max_concurrent_physical_gpus": max_concurrent,
        "allowed_physical_gpu_ids": list(range(6)), "reserved_physical_gpu_ids": [6, 7],
    }
    atomic_json(scheduling / f"{mode}_execution_summary.json", summary)
    event(events_path, "scheduler_finished", **summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    run_queue(args.revision_run.resolve(), args.mode)


if __name__ == "__main__":
    main()
