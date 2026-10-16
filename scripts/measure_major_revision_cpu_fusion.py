#!/usr/bin/env python3
"""Measure cached-vector fusion separately from visual scoring."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import json
import time
import numpy as np
from finconrank.strict_nested import atomic_csv, atomic_json, read_jsonl
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.revision_run
    output = run / "results/cost_benchmark"
    samples = read_jsonl(output / "locked_timing_queries.jsonl")
    predictions = {(r["language"], r["query_id"]): r for r in read_jsonl(run / "results/fusion_ablations_deduplicated_v2/query_level_evaluated_predictions.jsonl")}
    rows = []
    for sample in samples:
        row = predictions[(sample["language"], sample["query_id"])]
        config = json.loads((run / "results/fusion_ablations_deduplicated_v2/outer_configs" / f"{sample['language']}_outer_{sample['outer_fold']}.json").read_text())
        for name, system in row["systems"].items():
            candidates = system["candidates"]
            if not candidates or "fusion_components" not in candidates[0]:
                continue
            vector = np.asarray([item["fusion_components"] for item in candidates])
            # Timing arithmetic and sorting of cached normalized vectors only.
            if name == "e1_fixed_four_score":
                weights = np.asarray(config[name]["fusion"]["weights_colq_mono_fincon_metric"])
            else:
                weights = np.asarray(config[name]["weights"])
            repetitions = 1000
            _ = np.argsort(-(vector @ weights), kind="stable")
            started = time.perf_counter_ns()
            for _ in range(repetitions):
                _ = np.argsort(-(vector @ weights), kind="stable")
            seconds = (time.perf_counter_ns() - started) / 1e9 / repetitions
            rows.append({"language": sample["language"], "query_id": sample["query_id"], "system": name,
                         "candidate_count": len(candidates), "components": vector.shape[1], "repetitions": repetitions,
                         "cpu_vector_fusion_seconds": seconds})
    atomic_csv(output / "cpu_cached_vector_fusion_actual_weights.csv", rows)
    atomic_json(output / "cpu_fusion_actual_weights_boundary.json", {
        "rows": len(rows), "boundary": "Cached normalized-vector arithmetic and stable sorting with frozen outer-selected weights. Excludes OCR/metric extraction, normalization, model scoring, and service overhead; not complete-system end-to-end latency. Earlier uniform-weight diagnostic retained separately and superseded for deployment timing.",
    })


if __name__ == "__main__":
    main()
