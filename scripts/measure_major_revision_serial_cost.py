#!/usr/bin/env python3
"""Actual two-model serial cost on the already locked development sample."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import gc
import os
import time
from collections import defaultdict
from pathlib import Path
import torch
from finconrank.strict_nested import atomic_csv, atomic_json, read_json, read_jsonl, sha256_file, utc_now
from major_revision_gpu_scheduler import parse_inventory, select_stable_gpus
from run_major_revision_cost_benchmark import load_adapter, score_query, MONO_MAX_PIXELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    output = args.revision_run / "results/cost_benchmark"
    protocol = read_json(output / "protocol_lock.json")
    rows = read_jsonl(Path(protocol["sample_path"]))
    atomic_json(output / "serial_cost_protocol_addendum.json", {
        "locked_at": utc_now(), "sample_sha256": sha256_file(Path(protocol["sample_path"])),
        "script_sha256": sha256_file(Path(__file__)), "route": "Mono then corresponding outer FinConRank; batch2; fp16; max_pixels602112",
        "selection": "Same locked 34 development queries; no qrels or effectiveness read",
        "timing": "Resident two-model serialized scoring; full image decode/preprocess repeated per model; CUDA synchronized; load separately; frozen CPU calibration joined separately",
    })
    snapshots = []
    while True:
        snapshots.append(parse_inventory())
        choices = select_stable_gpus(snapshots, set(), 22000, requested=1)
        if choices:
            break
        time.sleep(10)
    gpu = choices[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    os.environ["HF_HUB_OFFLINE"] = "1"
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["finconrank_adapter"]].append(row)
    started = time.perf_counter()
    mp, mono, mt, mono_load = load_adapter(protocol["base_model"], protocol["mono_adapter"], torch.float16, None, MONO_MAX_PIXELS)
    load_total = mono_load
    measurements = []
    torch.cuda.reset_peak_memory_stats()
    for adapter, subset in sorted(grouped.items()):
        fp, fin, ft, loading = load_adapter(protocol["base_model"], adapter, torch.float16, None, MONO_MAX_PIXELS)
        load_total += loading
        for index, row in enumerate(subset):
            torch.cuda.synchronize()
            active = time.perf_counter()
            m = score_query(mp, mono, mt, row, "monoqwen2vl")
            f = score_query(fp, fin, ft, row, "finconrank")
            torch.cuda.synchronize()
            measurements.append({"language": row["language"], "query_id": row["query_id"], "outer_fold": row["outer_fold"],
                                 "warmup": index == 0, "candidate_count": len(row["candidates"]),
                                 "serial_visual_wall_seconds": time.perf_counter() - active,
                                 "mono_gpu_seconds": m["gpu_inference_seconds"], "fin_gpu_seconds": f["gpu_inference_seconds"],
                                 "physical_gpu_id": gpu.index})
        del fp, fin
        gc.collect()
        torch.cuda.empty_cache()
    atomic_csv(output / "serial_resident_visual_timing.csv", measurements)
    atomic_json(output / "serial_resident_visual_summary.json", {
        "status": "complete", "finished_at": utc_now(), "physical_gpu_id": gpu.index, "gpu_uuid": gpu.uuid,
        "queries": len(measurements), "pairs_per_model": sum(row["candidate_count"] for row in measurements),
        "load_seconds": load_total, "wall_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "boundary": "Actual two-model serial visual timing; add separately measured frozen CPU calibration for component-based complete-system cost, not network/service SLA",
    })


if __name__ == "__main__":
    main()
