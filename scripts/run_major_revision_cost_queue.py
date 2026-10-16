#!/usr/bin/env python3
"""Run the locked timing routes sequentially on an uncontended GPU."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from major_revision_gpu_scheduler import atomic_json, parse_inventory, select_stable_gpus, utc_now
from finconrank.strict_nested import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.revision_run.resolve()
    output = run / "results/cost_benchmark"
    script = Path(__file__).with_name("run_major_revision_cost_benchmark.py")
    addendum = output / "implementation_addendum.json"
    if not addendum.exists():
        atomic_json(addendum, {
            "created_at": utc_now(), "script_sha256": sha256_file(script),
            "original_protocol_sha256": sha256_file(output / "protocol_lock.json"),
            "change": "Release caller references before loading another adapter; no sampling, model, scoring or timing configuration changes.",
        })
    for route in ("monoqwen2vl", "finconrank", "docrerank"):
        if (output / f"{route}_summary.json").exists():
            continue
        snapshots = []
        while True:
            snapshots.append(parse_inventory())
            choices = select_stable_gpus(snapshots, set(), 18432, requested=1)
            if choices:
                gpu = choices[0]
                break
            time.sleep(20)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu.index),
                   FINCONRANK_PHYSICAL_GPU_ID=str(gpu.index), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        atomic_json(output / "queue_state.json", {"route": route, "physical_gpu_id": gpu.index, "started_at": utc_now(), "pid": os.getpid()})
        with (run / "logs" / f"cost_{route}.log").open("a") as log:
            result = subprocess.run([sys.executable, str(script), "benchmark", "--revision-run", str(run), "--route", route], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{route} failed: {result.returncode}")
    atomic_json(output / "queue_state.json", {"status": "complete", "finished_at": utc_now(), "pid": os.getpid()})


if __name__ == "__main__":
    main()
