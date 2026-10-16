#!/usr/bin/env python3
"""Read-only distribution and historical cost extraction for revision A2/E4."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import json
from collections import Counter
from pathlib import Path
from finconrank.strict_nested import atomic_csv, atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    rows, counts, sources = [], [], []
    for task in sorted((args.strict_run / "tasks").iterdir()):
        state_path = task / "state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        pairs_path = task / "training_data/train_pairs.jsonl"
        pairs = [json.loads(line) for line in pairs_path.open()]
        groups = {row["group_id"] for row in pairs}
        qids = {row["query_id"] for row in pairs}
        types = Counter(label for row in pairs for label in row.get("negative_types", []))
        stages = Counter(row.get("curriculum_stage", "unknown") for row in pairs)
        rows.append({"task_id": task.name, "language": state["language"], "kind": state["kind"],
                     "outer_fold": state["outer_fold"], "inner_fold": state.get("inner_fold"),
                     "pairs": len(pairs), "queries": len(qids), "report_groups": len(groups),
                     "training_seconds": state["training"]["seconds"],
                     "inference_seconds": state["inference"]["inference_wall_seconds_this_attempt"],
                     "micro_batch_size": state["micro_batch_size"], "gradient_accumulation": state["gradient_accumulation"],
                     "pair_source_sha256": sha256_file(pairs_path)})
        for dimension, counter in (("negative_type_multilabel", types), ("curriculum_stage", stages)):
            for label, count in sorted(counter.items(), key=lambda item: str(item[0])):
                counts.append({"task_id": task.name, "language": state["language"], "dimension": dimension,
                               "label": label, "count": count, "pair_denominator": len(pairs), "fraction": count / len(pairs)})
        for path in (state_path, pairs_path):
            sources.append({"source_path": str(path.resolve()), "sha256": sha256_file(path)})
    output = args.revision_run / "results/cached_diagnostics"
    atomic_csv(output / "a2_training_units.csv", rows)
    atomic_csv(output / "a2_training_conflict_distribution.csv", counts)
    atomic_json(output / "historical_training_cost_audit.json", {
        "units": len(rows), "inner_units": sum(row["kind"] == "inner" for row in rows),
        "outer_units": sum(row["kind"] == "outer" for row in rows),
        "summed_training_gpu_hours": sum(row["training_seconds"] for row in rows) / 3600,
        "summed_inference_gpu_hours": sum(row["inference_seconds"] for row in rows) / 3600,
        "boundary": "Recorded successful-attempt task durations, not elapsed wall time or exhaustive retry/selection cost; repeated CV pairs are not unique observations. Conflict types are automatic multi-label sampling metadata, not independent truth.",
        "sources": sources,
    })


if __name__ == "__main__":
    main()
