#!/usr/bin/env python3
"""Summarize route timing and explicitly bounded composite costs."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from finconrank.strict_nested import atomic_csv, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    output = args.revision_run / "results/cost_benchmark"
    measures, summaries = {}, []
    for route in ("monoqwen2vl", "finconrank", "docrerank"):
        with (output / f"{route}_query_timing.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        measures[route] = {(r["language"], r["query_id"]): r for r in rows}
        summary = json.loads((output / f"{route}_summary.json").read_text())
        for language in ("en", "zh"):
            subset = [r for r in rows if r["language"] == language and r["warmup"] == "False"]
            values = [float(r["query_active_seconds"]) for r in subset]
            summaries.append({"route": route, "language": language, "measured_queries_excluding_warmup": len(subset),
                              "mean_query_seconds": float(np.mean(values)), "median_query_seconds": float(np.median(values)),
                              "p95_query_seconds": float(np.quantile(values, .95)),
                              "queries_per_second": len(values) / sum(values), "peak_vram_gib": summary["peak_vram_gib"],
                              "model_loading_total_seconds": summary["model_load_seconds"],
                              "boundary": "Sequential offline active time; loading separate; no service SLA"})
    composites = []
    for key, mono in measures["monoqwen2vl"].items():
        fincon = measures["finconrank"][key]
        composites.append({"language": key[0], "query_id": key[1],
                           "mono_active_seconds": mono["query_active_seconds"],
                           "serial_mono_fincon_active_seconds": float(mono["query_active_seconds"]) + float(fincon["query_active_seconds"]),
                           "boundary": "Sum of separate real scoring measurements, not a jointly measured online cascade; repeated image decode/preparation included; metric evidence and normalization overhead excluded"})
    atomic_csv(output / "latency_throughput_summary.csv", summaries)
    atomic_csv(output / "composite_scoring_cost.csv", composites)
    atomic_json(output / "cost_completion_boundary.json", {
        "gpu_routes_measured": 3, "sample_queries": 34, "candidate_pairs_per_route": 680,
        "cpu_fusion": "actual frozen weights on normalized cached vectors; excludes metric extraction and normalization",
        "not_measured": ["single-process resident complete-system online wall latency", "OCR/metric feature construction latency", "production I/O and concurrency"],
        "interpretation": "Use real component timings and explicitly labeled sums, not an end-to-end service benchmark or exact historical optimized implementation cost.",
    })


if __name__ == "__main__":
    main()
