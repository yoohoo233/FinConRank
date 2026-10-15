#!/usr/bin/env python3
"""Resume strict nested CV with a hard cap on concurrent GPU tasks."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import finconrank_strict_nested_scheduler as base
from finconrank.strict_nested import atomic_json, read_json, utc_now


def save_progress(run_dir: Path, manifest: dict[str, Any], active: list[dict[str, Any]], limit: int) -> None:
    inner = [task for task in manifest["tasks"] if task["kind"] == "inner"]
    outer = [task for task in manifest["tasks"] if task["kind"] == "outer"]
    payload = read_json(run_dir / "progress.json")
    payload.update({
        "updated_at": utc_now(),
        "scheduler_pid": os.getpid(),
        "status": "running",
        "phase": "inner" if sum(task["status"] == "complete" for task in inner) < 40 else "outer",
        "max_concurrent_gpu_tasks": limit,
        "resource_limit_reason": "user requested that other GPUs be released",
        "inner_tasks": {
            "total": 40,
            **{status: sum(task["status"] == status for task in inner) for status in ("pending", "running", "complete", "failed_permanent")},
        },
        "outer_tasks": {
            "total": 10,
            **{status: sum(task["status"] == status for task in outer) for status in ("pending", "running", "complete", "failed_permanent")},
        },
        "running": {
            task["task_id"]: {"pid": task.get("worker_pid"), "gpu": task.get("physical_gpu_id")}
            for task in active
        },
    })
    atomic_json(run_dir / "progress.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-concurrent-tasks", type=int, default=2)
    parser.add_argument("--allowed-gpus", default="0,1,2,3,4,5")
    args = parser.parse_args()
    if args.max_concurrent_tasks < 1:
        raise ValueError("max-concurrent-tasks must be positive")
    allowed_gpus = {int(value) for value in args.allowed_gpus.split(",") if value.strip()}
    if not allowed_gpus:
        raise ValueError("allowed-gpus must contain at least one physical GPU ID")
    run_dir = args.run_dir.resolve()
    lock = base.acquire_scheduler_lock(run_dir)
    idle_streak = {gpu: 0 for gpu in allowed_gpus}
    last_gpu_log = 0.0
    try:
        while True:
            manifest = read_json(run_dir / "task_manifest.json")
            active: list[dict[str, Any]] = []
            for task in manifest["tasks"]:
                if task["status"] != "running":
                    continue
                state = base.read_task_state(run_dir, task["task_id"])
                pid = int(task.get("worker_pid", -1))
                if state and state.get("status") == "complete":
                    task["status"] = "complete"
                    task["finished_at"] = state["finished_at"]
                    base.archive_task_lock(run_dir, task["task_id"], "complete")
                elif base.pid_alive(pid):
                    active.append(task)
                else:
                    error = (state or {}).get("error", "task interrupted while reducing GPU concurrency")
                    task["last_error"] = error
                    base.archive_task_lock(run_dir, task["task_id"], "interrupted")
                    if int(task.get("attempts", 0)) >= base.MAX_ATTEMPTS:
                        task["status"] = "failed_permanent"
                    else:
                        task["status"] = "pending"
                        if "out of memory" in error.lower() or "cuda oom" in error.lower():
                            task["micro_batch_size"] = 1
                            task["gradient_accumulation"] = 4
                            task["inference_batch_size"] = 1
            base.save_manifest(run_dir, manifest)
            if any(task["status"] == "failed_permanent" for task in manifest["tasks"]):
                save_progress(run_dir, manifest, active, args.max_concurrent_tasks)
                raise RuntimeError("one or more tasks failed after three attempts")

            inner = [task for task in manifest["tasks"] if task["kind"] == "inner"]
            if len(inner) == 40 and all(task["status"] == "complete" for task in inner):
                base.run_selection(run_dir)
                manifest = read_json(run_dir / "task_manifest.json")
                active = [task for task in manifest["tasks"] if task["status"] == "running" and base.pid_alive(int(task.get("worker_pid", -1)))]
            outer = [task for task in manifest["tasks"] if task["kind"] == "outer"]
            if len(outer) == 10 and all(task["status"] == "complete" for task in outer):
                base.write_execution_artifacts(run_dir, manifest)
                base.freeze_outer_predictions(run_dir, manifest)
                save_progress(run_dir, manifest, active, args.max_concurrent_tasks)
                base.run_evaluation(run_dir)
                payload = read_json(run_dir / "progress.json")
                payload.update({"phase": "complete", "status": "complete", "outer_test_metrics_computed": True, "updated_at": utc_now()})
                atomic_json(run_dir / "progress.json", payload)
                break

            inventory = {
                gpu: info
                for gpu, info in base.gpu_inventory().items()
                if gpu in allowed_gpus
            }
            now = time.time()
            assigned = {int(task["physical_gpu_id"]) for task in active}
            for gpu, info in inventory.items():
                eligible = (
                    gpu not in assigned
                    and not info["compute_processes"]
                    and info["utilization_percent"] < base.MAX_IDLE_UTIL
                    and info["memory_free_mib"] >= base.MIN_FREE_MIB
                )
                idle_streak[gpu] = idle_streak[gpu] + 1 if eligible else 0
            if now - last_gpu_log >= base.GPU_LOG_SECONDS:
                with (run_dir / "gpu_monitor_limited.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "timestamp": utc_now(),
                        "max_concurrent_tasks": args.max_concurrent_tasks,
                        "available_ram_gib": base.available_ram_gib(),
                        "gpus": list(inventory.values()),
                        "idle_streak": idle_streak,
                        "active_tasks": [{"task_id": task["task_id"], "pid": task.get("worker_pid"), "gpu": task.get("physical_gpu_id")} for task in active],
                    }) + "\n")
                last_gpu_log = now

            pending = [task for task in manifest["tasks"] if task["status"] == "pending"]
            slots = max(0, args.max_concurrent_tasks - len(active))
            for gpu in sorted(inventory):
                if slots == 0 or not pending:
                    break
                if idle_streak[gpu] < 2 or gpu in assigned:
                    continue
                if base.available_ram_gib() < base.RAM_RESERVE_GIB + base.RAM_PER_TASK_GIB * (len(active) + 1):
                    break
                task = pending.pop(0)
                process = base.launch_task(run_dir, task, gpu)
                task["worker_pid"] = process.pid
                active.append(task)
                assigned.add(gpu)
                idle_streak[gpu] = 0
                slots -= 1
                base.save_manifest(run_dir, manifest)
            save_progress(run_dir, manifest, active, args.max_concurrent_tasks)
            time.sleep(base.POLL_SECONDS)
    finally:
        if lock.exists():
            os.replace(lock, run_dir / f"scheduler.limited.{os.getpid()}.finished.lock")


if __name__ == "__main__":
    main()
