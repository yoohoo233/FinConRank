#!/usr/bin/env python3
"""Generate numerical evidence index directly from frozen result tables."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
from pathlib import Path
from finconrank.strict_nested import atomic_json, utc_now


def table(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            try:
                if key not in ("query_count", "comparison_index", "seed", "report_group_count"):
                    value = f"{float(value):.6f}"
            except ValueError:
                pass
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.revision_run
    fusion = run / "results/fusion_ablations_deduplicated_v2"
    configs = [json.loads(path.read_text()) for path in sorted((fusion / "outer_configs").glob("*.json"))]
    zero = {lang: sum(c["e2_selected_zero_finconrank_weight"] for c in configs if c["language"] == lang) for lang in ("en", "zh")}
    workers = [json.loads(path.read_text()) for path in (run / "results/docrerank/workers").glob("*.stats.json")]
    summary = {"created_at": utc_now(), "zero_fincon_weight_folds": zero, "folds_per_language": 5,
               "docrerank_full_queries": sum(x["queries"] for x in workers),
               "docrerank_full_pairs": sum(x["scored_pairs"] for x in workers),
               "docrerank_summed_gpu_inference_hours": sum(x["gpu_inference_seconds"] for x in workers) / 3600,
               "docrerank_summed_worker_wall_hours": sum(x["worker_wall_seconds"] for x in workers) / 3600,
               "training_changed": False, "new_base_downloads": 0}
    atomic_json(run / "final_validation/numerical_evidence_summary.json", summary)
    sections = ["# Authorized Revision Results\n", "## E1/E2: Frozen Mono Top-20 Pool\n", table(fusion / "metrics_summary.csv"),
                f"\nZero FinConRank weight selected in English {zero['en']}/5 and Chinese {zero['zh']}/5 outer configurations. Selection used current-outer cross-fitted inner OOF only.\n",
                "### Separate Eight-Test Revision Family\n", table(fusion / "statistical_significance.csv"),
                "## E3: Frozen ColQwen2 Top-20 Pool\n", table(run / "results/docrerank/metrics_summary.csv"),
                "### Separate Eight-Test Direct-Reranker Family\n", table(run / "results/docrerank/statistical_significance.csv"),
                "## Costs\n", table(run / "results/cost_benchmark/latency_throughput_summary.csv"),
                "\nCandidate pools differ between E1/E2 and E3; do not subtract their metrics as same-pool effects. Cost measurements use warm cached evidence, exclude warmups as recorded, and separate loading. Composite route times are explicitly labeled sums, not a production service benchmark.\n",
                "## Limits\n\nNo retraining, extra model downloads, paper changes or reviewer responses. H1 second-annotator agreement remains unavailable. T1/T2 training requires separate approval. First-round 191/8/1 is stratified sample quality, not training-label accuracy. v1 duplicate-ID failure is superseded and never a final result.\n"]
    (run / "results_summary.md").write_text("\n".join(sections))


if __name__ == "__main__":
    main()
