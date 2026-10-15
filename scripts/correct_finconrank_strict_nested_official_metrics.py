#!/usr/bin/env python3
"""Correct post-evaluation metrics to official unique-doc pytrec semantics.

This is a deterministic reporting correction. It does not retrain a model,
change an outer-fold configuration, or alter frozen GPU score files.
"""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

from finconrank.release_paths import release_path

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytrec_eval

from evaluate_finconrank_strict_nested import SYSTEMS, clustered_test, holm
from finconrank.common import METRICS
from finconrank.strict_nested import (
    SEED,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)


ROOT_OUTPUTS = (
    "strict_nested_outer_query_predictions.jsonl",
    "metrics_summary.csv",
    "fold_metrics.csv",
    "delta_vs_mono.csv",
    "query_type_metrics.csv",
    "statistical_significance.csv",
    "statistical_significance.json",
    "metric_recomputation_audit.json",
    "leakage_audit.json",
    "completion_audit.json",
    "OUTER_TEST_EVALUATION_COMPLETE",
)

DOC_OUTPUTS = (
    "finconrank_strict_nested_experiment_report.md",
    "finconrank_strict_nested_leakage_audit.md",
    "finconrank_strict_nested_error_analysis.md",
    "finconrank_strict_nested_completion_audit.md",
)


def unique_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    result = []
    for item in candidates:
        corpus_id = str(item["corpus_id"])
        if corpus_id in seen:
            continue
        seen.add(corpus_id)
        copy = dict(item)
        copy["rank"] = len(result) + 1
        result.append(copy)
    return result, len(candidates) - len(result)


def evaluate_official(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> dict[str, float]:
    relevant = {str(doc_id): int(rel) for doc_id, rel in qrels.items() if int(rel) > 0}
    ranked = [str(item["corpus_id"]) for item in candidates]

    def dcg(gains: list[int]) -> float:
        return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))

    output: dict[str, float] = {}
    for cutoff in (5, 10):
        gains = [relevant.get(doc_id, 0) for doc_id in ranked[:cutoff]]
        ideal = sorted(relevant.values(), reverse=True)[:cutoff]
        denominator = dcg(ideal)
        output[f"nDCG@{cutoff}"] = dcg(gains) / denominator if denominator else 0.0
        output[f"Recall@{cutoff}"] = (
            len(set(ranked[:cutoff]) & set(relevant)) / len(relevant) if relevant else 0.0
        )
    output["MRR@10"] = next(
        (1.0 / rank for rank, doc_id in enumerate(ranked[:10], 1) if doc_id in relevant),
        0.0,
    )
    return output


def aggregate(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in rows]))
        if rows else 0.0
        for metric in METRICS
    }


def pytrec_check(candidates: list[dict[str, Any]], qrels: dict[str, int], metrics: dict[str, float]) -> float:
    qrel = {"q": {str(doc_id): int(rel) for doc_id, rel in qrels.items() if int(rel) > 0}}
    count = len(candidates)
    run = {"q": {str(item["corpus_id"]): float(count - rank) for rank, item in enumerate(candidates)}}
    result = pytrec_eval.RelevanceEvaluator(qrel, {"ndcg_cut_5", "ndcg_cut_10", "recall_5", "recall_10"}).evaluate(run)["q"]
    pairs = {
        "nDCG@5": result["ndcg_cut_5"],
        "nDCG@10": result["ndcg_cut_10"],
        "Recall@5": result["recall_5"],
        "Recall@10": result["recall_10"],
    }
    return max(abs(metrics[name] - value) for name, value in pairs.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(release_path('FINCONRANK_WORKSPACE', '.', '')))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    project = args.project_dir.resolve()
    marker = run_dir / "OFFICIAL_METRIC_CORRECTION_COMPLETE"
    if marker.exists():
        print(marker.read_text(encoding="utf-8"))
        return

    timestamp = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
    backup = run_dir / f"pre_official_metric_correction_{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    source_hashes: dict[str, str] = {}
    for name in ROOT_OUTPUTS:
        source = run_dir / name
        if source.exists():
            source_hashes[name] = sha256_file(source)
            shutil.copy2(source, backup / name)
    docs_backup = backup / "docs"
    docs_backup.mkdir()
    for name in DOC_OUTPUTS:
        source = project / "docs" / name
        if source.exists():
            source_hashes[f"docs/{name}"] = sha256_file(source)
            shutil.copy2(source, docs_backup / name)
    atomic_json(backup / "snapshot_manifest.json", {
        "created_at": utc_now(),
        "reason": "Preserve first post-evaluation outputs before official unique-document metric correction.",
        "files": source_hashes,
    })

    rows = read_jsonl(backup / "strict_nested_outer_query_predictions.jsonl")
    duplicate_audit = []
    max_pytrec_error = 0.0
    for row in rows:
        for system in SYSTEMS:
            candidates, removed = unique_candidates(row["systems"][system]["candidates"])
            metrics = evaluate_official(candidates, row["qrels"])
            max_pytrec_error = max(max_pytrec_error, pytrec_check(candidates, row["qrels"], metrics))
            if removed:
                duplicate_audit.append({
                    "language": row["language"],
                    "outer_fold": row["outer_fold"],
                    "query_id": row["query_id"],
                    "system": system,
                    "candidate_rows_before": len(row["systems"][system]["candidates"]),
                    "unique_candidates_after": len(candidates),
                    "duplicate_rows_removed": removed,
                    "metrics_before": row["systems"][system]["metrics"],
                    "metrics_after": metrics,
                })
            row["systems"][system]["candidates"] = candidates
            row["systems"][system]["metrics"] = metrics

    atomic_jsonl(run_dir / "strict_nested_outer_query_predictions.jsonl", rows)
    prediction_sha = sha256_file(run_dir / "strict_nested_outer_query_predictions.jsonl")
    summary = []
    fold_rows = []
    query_type_rows = []
    for language in ("en", "zh"):
        language_rows = [row for row in rows if row["language"] == language]
        for system in SYSTEMS:
            summary.append({"language": language, "run": system, "query_count": len(language_rows), **aggregate(language_rows, system)})
        for outer_fold in range(5):
            fold = [row for row in language_rows if int(row["outer_fold"]) == outer_fold]
            for system in SYSTEMS:
                fold_rows.append({"language": language, "outer_fold": outer_fold, "run": system, "query_count": len(fold), **aggregate(fold, system)})
        for query_type in sorted({row["query_type"] for row in language_rows}):
            subset = [row for row in language_rows if row["query_type"] == query_type]
            for system in SYSTEMS:
                query_type_rows.append({"language": language, "query_type": query_type, "run": system, "query_count": len(subset), **aggregate(subset, system)})
    atomic_csv(run_dir / "metrics_summary.csv", summary)
    atomic_csv(run_dir / "fold_metrics.csv", fold_rows)
    atomic_csv(run_dir / "query_type_metrics.csv", query_type_rows)

    deltas = []
    significance = []
    for language in ("en", "zh"):
        language_summary = [row for row in summary if row["language"] == language]
        mono = next(row for row in language_summary if row["run"] == "monoqwen2vl")
        for row in language_summary:
            deltas.append({
                "language": language, "run": row["run"], "baseline": "monoqwen2vl",
                **{f"delta_{metric}": row[metric] - mono[metric] for metric in METRICS},
            })
        language_rows = [row for row in rows if row["language"] == language]
        for baseline in ("monoqwen2vl", "mono_metric_aware"):
            for metric in METRICS:
                query_deltas = [
                    row["systems"]["selected_complete_system"]["metrics"][metric]
                    - row["systems"][baseline]["metrics"][metric]
                    for row in language_rows
                ]
                significance.append({
                    "family": "primary" if metric in {"nDCG@10", "MRR@10"} else "exploratory",
                    "language": language,
                    "system": "selected_complete_system",
                    "baseline": baseline,
                    "metric": metric,
                    **clustered_test(query_deltas, [row["source_report_group"] for row in language_rows], SEED + len(significance) * 31),
                })
    atomic_csv(run_dir / "delta_vs_mono.csv", deltas)
    for family in ("primary", "exploratory"):
        holm([row for row in significance if row["family"] == family])
    atomic_csv(run_dir / "statistical_significance.csv", significance)
    atomic_json(run_dir / "statistical_significance.json", {"tests": significance})

    # A second independent pass from the corrected serialized predictions.
    reloaded = read_jsonl(run_dir / "strict_nested_outer_query_predictions.jsonl")
    max_recompute_error = 0.0
    for expected in summary:
        subset = [row for row in reloaded if row["language"] == expected["language"]]
        recomputed = {
            metric: float(np.mean([
                evaluate_official(row["systems"][expected["run"]]["candidates"], row["qrels"])[metric]
                for row in subset
            ]))
            for metric in METRICS
        }
        max_recompute_error = max(max_recompute_error, *(
            abs(recomputed[metric] - expected[metric]) for metric in METRICS
        ))

    correction = {
        "status": "complete",
        "created_at": utc_now(),
        "scope": "post-evaluation metric semantics only",
        "reason": "Official FinRAGBench-V load_from_trec represents each query run as a doc-id keyed dictionary before pytrec_eval; duplicate candidate rows must not count twice.",
        "model_or_checkpoint_changed": False,
        "split_or_outer_config_changed": False,
        "frozen_gpu_score_files_changed": False,
        "outer_test_parameter_selection_performed": False,
        "source_snapshot_dir": str(backup),
        "source_snapshot_manifest_sha256": sha256_file(backup / "snapshot_manifest.json"),
        "affected_unique_queries": len({item["query_id"] for item in duplicate_audit}),
        "affected_system_query_rows": len(duplicate_audit),
        "duplicate_audit": duplicate_audit,
        "pytrec_max_absolute_error": max_pytrec_error,
        "independent_recomputation_max_absolute_error": max_recompute_error,
        "corrected_prediction_sha256": prediction_sha,
    }
    atomic_json(run_dir / "metric_definition_correction.json", correction)
    atomic_json(run_dir / "metric_recomputation_audit.json", {
        "created_at": utc_now(),
        "prediction_sha256": prediction_sha,
        "metric_definition": "official unique-document run semantics validated against pytrec_eval",
        "maximum_absolute_metric_error": max(max_pytrec_error, max_recompute_error),
        "tolerance": 1e-12,
        "passed": max(max_pytrec_error, max_recompute_error) < 1e-12,
    })
    leakage = read_json(backup / "leakage_audit.json")
    leakage.update({
        "metric_definition_correction_applied": True,
        "metric_correction_changed_model_or_selection": False,
        "metric_correction_affected_unique_queries": correction["affected_unique_queries"],
        "passed": bool(leakage.get("passed")) and max(max_pytrec_error, max_recompute_error) < 1e-12,
    })
    atomic_json(run_dir / "leakage_audit.json", leakage)
    completion = read_json(backup / "completion_audit.json")
    completion.update({
        "status": "complete" if leakage["passed"] else "failed",
        "completed_at": utc_now(),
        "evaluated_prediction_sha256": prediction_sha,
        "metric_definition": "official unique-document pytrec semantics",
        "metric_definition_correction_sha256": sha256_file(run_dir / "metric_definition_correction.json"),
        "metric_recomputation_max_error": max(max_pytrec_error, max_recompute_error),
    })
    atomic_json(run_dir / "completion_audit.json", completion)
    (run_dir / "OUTER_TEST_EVALUATION_COMPLETE").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({
        "completed_at": utc_now(),
        "correction_path": str((run_dir / "metric_definition_correction.json").resolve()),
        "corrected_prediction_sha256": prediction_sha,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(correction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
