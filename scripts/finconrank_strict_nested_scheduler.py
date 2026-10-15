#!/usr/bin/env python3
"""Recoverable dynamic GPU scheduler for strict FinConRank nested CV."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from finconrank.strict_nested import (
    LANGUAGES,
    OUTER_FOLDS,
    PROJECT,
    atomic_csv,
    atomic_json,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    task_dir,
    task_id,
    utc_now,
)


POLL_SECONDS = 60
GPU_LOG_SECONDS = 300
MIN_FREE_MIB = 20000
MAX_IDLE_UTIL = 10
RAM_PER_TASK_GIB = 24
RAM_RESERVE_GIB = 16
MAX_ATTEMPTS = 3
BOOTSTRAP_TASK = "inner_en_outer0_inner0"


def parse_csv(command: list[str]) -> list[list[str]]:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]


def gpu_inventory() -> dict[int, dict[str, Any]]:
    rows = parse_csv([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus = {
        int(row[0]): {
            "physical_gpu_id": int(row[0]), "uuid": row[1], "name": row[2],
            "memory_total_mib": int(row[3]), "memory_used_mib": int(row[4]),
            "memory_free_mib": int(row[5]), "utilization_percent": int(row[6]),
            "compute_processes": [],
        }
        for row in rows if 0 <= int(row[0]) <= 7
    }
    uuid_to_index = {item["uuid"]: index for index, item in gpus.items()}
    try:
        processes = parse_csv([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
    except subprocess.CalledProcessError:
        processes = []
    for row in processes:
        if row[0] in uuid_to_index:
            gpus[uuid_to_index[row[0]]]["compute_processes"].append({
                "pid": int(row[1]), "process_name": row[2], "used_memory_mib": int(row[3]),
            })
    return gpus


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable not found")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def acquire_scheduler_lock(run_dir: Path) -> Path:
    lock = run_dir / "scheduler.lock"
    payload = {"pid": os.getpid(), "started_at": utc_now()}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        existing = read_json(lock)
        if pid_alive(int(existing.get("pid", -1))):
            raise RuntimeError(f"scheduler already running: {existing}")
        stale = run_dir / f"scheduler.stale.{int(time.time())}.lock"
        os.replace(lock, stale)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return lock


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_json(run_dir / "task_manifest.json", manifest)


def task_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task_id"]: task for task in manifest["tasks"]}


def task_lock(run_dir: Path, identifier: str, attempt: int) -> Path:
    directory = run_dir / "task_locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"task_id": identifier, "attempt": attempt, "scheduler_pid": os.getpid(), "created_at": utc_now()}, handle)
        handle.write("\n")
    return path


def archive_task_lock(run_dir: Path, identifier: str, suffix: str) -> None:
    path = run_dir / f"task_locks/{identifier}.lock"
    if path.exists():
        os.replace(path, run_dir / f"task_locks/{identifier}.{suffix}.{int(time.time())}.lock")


def launch_task(run_dir: Path, task: dict[str, Any], gpu: int) -> subprocess.Popen:
    attempt = int(task["attempts"]) + 1
    task_lock(run_dir, task["task_id"], attempt)
    task["attempts"] = attempt
    task["status"] = "running"
    task["physical_gpu_id"] = gpu
    task["started_at"] = utc_now()
    log_path = run_dir / f"logs/{task['task_id']}_attempt{attempt}.log"
    task["log_path"] = str(log_path.resolve())
    command = [
        sys.executable, str(PROJECT / "scripts/run_finconrank_strict_nested_task.py"),
        "--run-dir", str(run_dir), "--task-id", task["task_id"],
        "--attempt", str(attempt), "--physical-gpu-id", str(gpu),
        "--micro-batch-size", str(task["micro_batch_size"]),
        "--gradient-accumulation", str(task["gradient_accumulation"]),
        "--inference-batch-size", str(task.get("inference_batch_size", 2)),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "src:.",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_ENDPOINT": "https://hf-mirror.com",
    })
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(json.dumps({"event": "scheduler_launch", "at": utc_now(), "command": command, "physical_gpu_id": gpu}) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command, cwd=PROJECT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    task["worker_pid"] = process.pid
    task["command"] = command
    log_handle.close()
    return process


def persisted_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(manifest))


def read_task_state(run_dir: Path, identifier: str) -> dict[str, Any] | None:
    path = task_dir(run_dir, identifier) / "state.json"
    return read_json(path) if path.exists() else None


def progress_summary(manifest: dict[str, Any], running: dict[str, subprocess.Popen]) -> dict[str, Any]:
    counts = {"inner": {}, "outer": {}}
    for kind in counts:
        values = [task for task in manifest["tasks"] if task["kind"] == kind]
        counts[kind] = {
            "total": 40 if kind == "inner" else 10,
            **{status: sum(task["status"] == status for task in values) for status in ("pending", "running", "complete", "failed_permanent")},
        }
    return {
        "phase": "inner" if counts["inner"]["complete"] < 40 else "outer" if counts["outer"]["complete"] < 10 else "evaluation",
        "inner_tasks": counts["inner"], "outer_tasks": counts["outer"],
        "running": {identifier: {"pid": process.pid, "gpu": next(task["physical_gpu_id"] for task in manifest["tasks"] if task["task_id"] == identifier)} for identifier, process in running.items()},
    }


def write_progress(run_dir: Path, manifest: dict[str, Any], running: dict[str, subprocess.Popen], extra: dict[str, Any] | None = None) -> None:
    previous = read_json(run_dir / "progress.json")
    previous.update(progress_summary(manifest, running))
    previous.update({"updated_at": utc_now(), "scheduler_pid": os.getpid(), "status": "running"})
    if extra:
        previous.update(extra)
    atomic_json(run_dir / "progress.json", previous)


def run_selection(run_dir: Path) -> None:
    marker = run_dir / "OUTER_CONFIGS_LOCKED"
    if marker.exists():
        return
    command = [sys.executable, str(PROJECT / "scripts/select_finconrank_strict_nested.py"), "--run-dir", str(run_dir)]
    log = run_dir / "logs/inner_selection.log"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT, env={**os.environ, "PYTHONPATH": "src:."}, stdout=handle, stderr=subprocess.STDOUT, check=True)
    marker.write_text(json.dumps({"locked_at": utc_now(), "command": command}, indent=2) + "\n", encoding="utf-8")


def freeze_outer_predictions(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "outer_prediction_freeze_manifest.json"
    if path.exists():
        return read_json(path)
    records = []
    for language in LANGUAGES:
        for outer in range(OUTER_FOLDS):
            identifier = task_id("outer", language, outer)
            task = task_map(manifest)[identifier]
            state = read_task_state(run_dir, identifier)
            if task["status"] != "complete" or not state or state["status"] != "complete":
                raise RuntimeError(f"cannot freeze incomplete outer task: {identifier}")
            prediction = Path(state["prediction_path"])
            split = read_json(run_dir / f"splits/{identifier}.json")
            rows = read_jsonl(prediction)
            if {row["query_id"] for row in rows} != set(split["test_qids"]):
                raise RuntimeError(f"outer qid coverage failure: {identifier}")
            if any(row["source_report_group"] in set(split["development_report_groups"]) for row in rows):
                raise RuntimeError(f"outer provenance group failure: {identifier}")
            records.append({
                "task_id": identifier, "language": language, "outer_fold": outer,
                "path": str(prediction.resolve()), "sha256": sha256_file(prediction),
                "bytes": prediction.stat().st_size, "query_count": len(rows),
                "candidate_count": sum(len(row["candidates"]) for row in rows),
                "checkpoint_path": state["checkpoint_path"], "checkpoint_sha256": state["checkpoint_sha256"],
            })
    freeze = {
        "status": "frozen_before_qrels_evaluation", "frozen_at": utc_now(),
        "prediction_count": len(records), "predictions": records,
        "outer_test_metrics_computed_at_freeze": False,
    }
    freeze["manifest_sha256"] = stable_hash({
        "protocol_sha256": sha256_file(run_dir / "protocol_lock.json"),
        "predictions": records,
    })
    atomic_json(path, freeze)
    return freeze


def write_execution_artifacts(run_dir: Path, manifest: dict[str, Any]) -> None:
    checkpoint_rows, coverage_rows, gpu_rows = [], [], []
    for task in manifest["tasks"]:
        state = read_task_state(run_dir, task["task_id"])
        if not state or state.get("status") != "complete":
            continue
        checkpoint_rows.append({
            "task_id": task["task_id"], "kind": task["kind"], "language": task["language"],
            "outer_fold": task["outer_fold"], "inner_fold": task.get("inner_fold"),
            "checkpoint_path": state["checkpoint_path"], "checkpoint_sha256": state["checkpoint_sha256"],
            "training_report_group_count": state["training"]["report_groups"],
            "training_inference_overlap_count": len(state["provenance"]["training_inference_report_group_overlap"]),
            "training_outer_test_overlap_count": len(state["provenance"]["training_outer_test_report_group_overlap"]),
        })
        inference = state["inference"]
        coverage_rows.append({
            "task_id": task["task_id"], "query_count": inference["query_count"],
            "candidate_count": inference["candidate_count"], "scored_candidate_count": inference["scored_candidate_count"],
            "missing_image_count": inference["missing_image_count"], "coverage": inference["coverage"],
        })
        gpu_rows.append({
            "task_id": task["task_id"], "physical_gpu_id": state["physical_gpu_id"],
            "attempt": state["attempt"], "training_seconds": state["training"]["seconds"],
            "inference_seconds": inference["inference_wall_seconds_this_attempt"],
            "training_peak_vram_gib": state["training"]["peak_vram_gib"],
            "inference_peak_vram_gib": inference["inference_peak_vram_gib"],
        })
    atomic_csv(run_dir / "checkpoint_provenance.csv", checkpoint_rows)
    atomic_csv(run_dir / "candidate_coverage.csv", coverage_rows)
    atomic_csv(run_dir / "gpu_execution_summary.csv", gpu_rows)
    atomic_json(run_dir / "gpu_execution_summary.json", {
        "created_at": utc_now(), "task_count": len(gpu_rows),
        "physical_gpu_ids": sorted({row["physical_gpu_id"] for row in gpu_rows}),
        "total_training_gpu_seconds": sum(float(row["training_seconds"]) for row in gpu_rows),
        "total_inference_gpu_seconds": sum(float(row["inference_seconds"]) for row in gpu_rows),
        "records": gpu_rows,
    })


def run_evaluation(run_dir: Path) -> None:
    command = [sys.executable, str(PROJECT / "scripts/evaluate_finconrank_strict_nested.py"), "--run-dir", str(run_dir)]
    log = run_dir / "logs/final_evaluation.log"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT, env={**os.environ, "PYTHONPATH": "src:."}, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    lock = acquire_scheduler_lock(run_dir)
    running: dict[str, subprocess.Popen] = {}
    gpu_for_task: dict[str, int] = {}
    idle_streak = {gpu: 0 for gpu in range(8)}
    last_gpu_log = 0.0
    bootstrap_complete = bool(read_json(run_dir / "progress.json").get("bootstrap_inner_unit_complete"))
    try:
        while True:
            manifest = read_json(run_dir / "task_manifest.json")
            tasks = task_map(manifest)
            # Recover or reconcile task processes after scheduler restart.
            for identifier, task in tasks.items():
                if task["status"] != "running" or identifier in running:
                    continue
                pid = int(task.get("worker_pid", -1))
                state = read_task_state(run_dir, identifier)
                if state and state.get("status") == "complete":
                    task["status"] = "complete"
                    task["finished_at"] = state["finished_at"]
                    archive_task_lock(run_dir, identifier, "complete")
                elif pid_alive(pid):
                    # A detached worker from a previous scheduler is left alone.
                    task["status"] = "running"
                else:
                    task["status"] = "pending"
                    archive_task_lock(run_dir, identifier, "stale")
            if tasks.get(BOOTSTRAP_TASK, {}).get("status") == "complete":
                bootstrap_complete = True
                eta_path = run_dir / "bootstrap_eta.json"
                if not eta_path.exists():
                    state = read_task_state(run_dir, BOOTSTRAP_TASK)
                    if state:
                        elapsed = float(state["training"]["seconds"]) + float(state["inference"]["inference_wall_seconds_this_attempt"])
                        atomic_json(eta_path, {
                            "completed_at": state["finished_at"], "task_id": BOOTSTRAP_TASK,
                            "single_gpu_unit_seconds": elapsed, "remaining_inner_units": 39,
                            "single_gpu_remaining_hours": 39 * elapsed / 3600.0,
                            "note": "Recovered ETA; actual wall time varies by language, pair/candidate count, and concurrency.",
                        })
            for identifier, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                task = tasks[identifier]
                state = read_task_state(run_dir, identifier)
                if code == 0 and state and state.get("status") == "complete":
                    task["status"] = "complete"
                    task["finished_at"] = state["finished_at"]
                    archive_task_lock(run_dir, identifier, "complete")
                    if identifier == BOOTSTRAP_TASK:
                        bootstrap_complete = True
                        elapsed = float(state["training"]["seconds"]) + float(state["inference"]["inference_wall_seconds_this_attempt"])
                        atomic_json(run_dir / "bootstrap_eta.json", {
                            "completed_at": utc_now(), "task_id": identifier,
                            "single_gpu_unit_seconds": elapsed,
                            "remaining_inner_units": 39,
                            "single_gpu_remaining_hours": 39 * elapsed / 3600.0,
                            "note": "Wall time varies with language, pair count, candidate count, and future free-GPU concurrency.",
                        })
                else:
                    error = (state or {}).get("error", f"worker exit code {code}")
                    task["last_error"] = error
                    archive_task_lock(run_dir, identifier, "failed")
                    if task["attempts"] < MAX_ATTEMPTS:
                        task["status"] = "pending"
                        if "out of memory" in error.lower() or "cuda oom" in error.lower():
                            task["micro_batch_size"] = 1
                            task["gradient_accumulation"] = 4
                            task["inference_batch_size"] = 1
                    else:
                        task["status"] = "failed_permanent"
                running.pop(identifier)
                gpu_for_task.pop(identifier, None)
            save_manifest(run_dir, persisted_manifest(manifest))
            if any(task["status"] == "failed_permanent" for task in manifest["tasks"]):
                write_progress(run_dir, manifest, running, {"status": "failed_permanent"})
                raise RuntimeError("one or more tasks failed after three attempts")
            inner = [task for task in manifest["tasks"] if task["kind"] == "inner"]
            if len(inner) == 40 and all(task["status"] == "complete" for task in inner):
                run_selection(run_dir)
                manifest = read_json(run_dir / "task_manifest.json")
                tasks = task_map(manifest)
            outer = [task for task in manifest["tasks"] if task["kind"] == "outer"]
            if len(outer) == 10 and all(task["status"] == "complete" for task in outer):
                write_execution_artifacts(run_dir, manifest)
                freeze_outer_predictions(run_dir, manifest)
                write_progress(run_dir, manifest, running, {"phase": "evaluation", "outer_predictions_frozen": True})
                run_evaluation(run_dir)
                write_progress(run_dir, manifest, running, {"phase": "complete", "status": "complete", "outer_test_metrics_computed": True})
                break
            inventory = gpu_inventory()
            now = time.time()
            assigned_gpus = set(gpu_for_task.values())
            for gpu, info in inventory.items():
                externally_busy = bool(info["compute_processes"]) and gpu not in assigned_gpus
                eligible = (
                    gpu not in assigned_gpus and not externally_busy
                    and info["utilization_percent"] < MAX_IDLE_UTIL
                    and info["memory_free_mib"] >= MIN_FREE_MIB
                )
                idle_streak[gpu] = idle_streak[gpu] + 1 if eligible else 0
            if now - last_gpu_log >= GPU_LOG_SECONDS:
                with (run_dir / "gpu_monitor.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "timestamp": utc_now(), "available_ram_gib": available_ram_gib(),
                        "gpus": list(inventory.values()), "idle_streak": idle_streak,
                        "running_tasks": {identifier: {"pid": process.pid, "gpu": gpu_for_task[identifier]} for identifier, process in running.items()},
                    }) + "\n")
                    handle.flush()
                last_gpu_log = now
            candidates = [task for task in manifest["tasks"] if task["status"] == "pending"]
            if not bootstrap_complete:
                candidates = [task for task in candidates if task["task_id"] == BOOTSTRAP_TASK]
            for gpu in sorted(inventory):
                if idle_streak[gpu] < 2 or gpu in assigned_gpus or not candidates:
                    continue
                if available_ram_gib() < RAM_RESERVE_GIB + RAM_PER_TASK_GIB * (len(running) + 1):
                    break
                task = candidates.pop(0)
                process = launch_task(run_dir, task, gpu)
                running[task["task_id"]] = process
                gpu_for_task[task["task_id"]] = gpu
                assigned_gpus.add(gpu)
                idle_streak[gpu] = 0
                save_manifest(run_dir, persisted_manifest(manifest))
            write_progress(run_dir, manifest, running, {"bootstrap_inner_unit_complete": bootstrap_complete})
            time.sleep(POLL_SECONDS)
    finally:
        # Do not terminate detached task processes on scheduler shutdown; they
        # remain recoverable and are reconciled on restart.
        if lock.exists():
            finished = run_dir / f"scheduler.{os.getpid()}.finished.lock"
            os.replace(lock, finished)


if __name__ == "__main__":
    main()
