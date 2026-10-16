#!/usr/bin/env python3
"""Conservative GPU inventory and lease helper for FinConRank major revision."""

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
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_PHYSICAL_GPUS = 4
PROJECT_ALLOWED_INDICES = frozenset(range(6))
RESERVED_INDICES = frozenset({6, 7})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class GPUStatus:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...] = ()
    process_visibility_known: bool = True
    scheduler_reserved: bool = False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def descendants(pid: int, parent_map: dict[int, int]) -> set[int]:
    found = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parent_map.items():
            if parent in found and child not in found:
                found.add(child)
                changed = True
    return found


def eligible(
    status: GPUStatus,
    active_goal_pids: set[int],
    required_free_mib: int,
    allowed_indices: frozenset[int] = PROJECT_ALLOWED_INDICES,
) -> tuple[bool, str]:
    if status.index not in allowed_indices or status.index in RESERVED_INDICES:
        return False, "project_not_authorized"
    if status.scheduler_reserved:
        return False, "scheduler_reserved"
    if not status.process_visibility_known:
        return False, "process_visibility_unknown"
    external = set(status.compute_pids) - active_goal_pids
    if external:
        return False, "external_compute_process"
    if set(status.compute_pids) & active_goal_pids:
        return False, "already_goal_leased"
    if status.utilization_percent > 5:
        return False, "utilization_above_5_percent"
    used_limit = max(1024, int(status.memory_total_mib * 0.05))
    if status.memory_used_mib > used_limit:
        return False, "memory_used_above_idle_threshold"
    if status.memory_free_mib < required_free_mib:
        return False, "insufficient_free_memory"
    return True, "eligible"


def select_stable_gpus(
    snapshots: list[list[GPUStatus]],
    active_goal_pids: set[int],
    required_free_mib: int,
    requested: int = MAX_PHYSICAL_GPUS,
    allowed_indices: frozenset[int] = PROJECT_ALLOWED_INDICES,
) -> list[GPUStatus]:
    if len(snapshots) < 3:
        return []
    by_uuid = [{gpu.uuid: gpu for gpu in snapshot} for snapshot in snapshots[-3:]]
    common = set(by_uuid[0]) & set(by_uuid[1]) & set(by_uuid[2])
    stable = []
    for uuid in common:
        observations = [mapping[uuid] for mapping in by_uuid]
        checks = [eligible(item, active_goal_pids, required_free_mib, allowed_indices)[0] for item in observations]
        if all(checks):
            stable.append(observations[-1])
    stable.sort(key=lambda item: (-item.memory_free_mib, item.index))
    return stable[: min(requested, MAX_PHYSICAL_GPUS)]


def parse_inventory() -> list[GPUStatus]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    pids: dict[str, list[int]] = {}
    visibility_known = process_query.returncode == 0
    if visibility_known:
        for line in process_query.stdout.splitlines():
            if not line.strip():
                continue
            uuid, pid = [value.strip() for value in line.split(",", 1)]
            try:
                pids.setdefault(uuid, []).append(int(pid))
            except ValueError:
                visibility_known = False
    output = []
    for line in gpu_query.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        index, uuid, name, total, used, free, utilization = values
        output.append(
            GPUStatus(
                index=int(index), uuid=uuid, name=name,
                memory_total_mib=int(total), memory_used_mib=int(used), memory_free_mib=int(free),
                utilization_percent=int(utilization), compute_pids=tuple(sorted(pids.get(uuid, []))),
                process_visibility_known=visibility_known, scheduler_reserved=False,
            )
        )
    return output


def read_leases(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"run_id": None, "updated_at": None, "leases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def live_goal_pids(leases: dict[str, Any]) -> set[int]:
    output = set()
    for lease in leases.get("leases", []):
        if lease.get("status") != "active":
            continue
        pid = int(lease["pid"])
        if Path(f"/proc/{pid}").exists():
            output.add(pid)
    return output


def acquire(
    lease_path: Path,
    run_id: str,
    task_id: str,
    pid: int,
    gpu: GPUStatus,
) -> dict[str, Any]:
    leases = read_leases(lease_path)
    active = [lease for lease in leases.get("leases", []) if lease.get("status") == "active"]
    active_uuids = {lease["gpu_uuid"] for lease in active}
    if gpu.uuid in active_uuids:
        raise RuntimeError(f"GPU already leased: {gpu.uuid}")
    if len(active_uuids | {gpu.uuid}) > MAX_PHYSICAL_GPUS:
        raise RuntimeError("major-revision goal would exceed four physical GPUs")
    lease = {
        "run_id": run_id, "task_id": task_id, "pid": pid,
        "physical_gpu_id": gpu.index, "gpu_uuid": gpu.uuid,
        "logical_cuda_device": 0, "started_at": utc_now(), "status": "active",
    }
    leases.update({"run_id": run_id, "updated_at": utc_now()})
    leases.setdefault("leases", []).append(lease)
    atomic_json(lease_path, leases)
    return lease


def release(lease_path: Path, task_id: str, status: str) -> None:
    leases = read_leases(lease_path)
    matched = False
    for lease in leases.get("leases", []):
        if lease.get("task_id") == task_id and lease.get("status") == "active":
            lease["status"] = status
            lease["finished_at"] = utc_now()
            matched = True
    if not matched:
        raise RuntimeError(f"active lease not found: {task_id}")
    leases["updated_at"] = utc_now()
    atomic_json(lease_path, leases)


def command_inventory(output_dir: Path) -> None:
    statuses = parse_inventory()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gpu_inventory.csv"
    rows = [asdict(status) for status in statuses]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"captured_at": utc_now(), "gpus": rows}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        command_inventory(args.output_dir.resolve())


if __name__ == "__main__":
    main()
