#!/usr/bin/env python3
"""Time actual frozen-parameter calibration on locked cost queries."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import time
import numpy as np
from pathlib import Path
from finconrank.strict_nested import INDEX_DIR, ROUTER_DIR, atomic_csv, read_json, read_jsonl, load_candidate_cache, task_id
from evaluate_finconrank import load_evidence, load_source_scores
from evaluate_finconrank_strict_nested import apply_metric, load_fin_scores
from select_finconrank_strict_nested import add_fincon, direct_candidates, fuse
from run_major_revision_fusion_ablations import deduplicate_candidates, metric_score_map, fuse_matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-run", type=Path, required=True)
    parser.add_argument("--strict-run", type=Path, required=True)
    args = parser.parse_args()
    run, strict = args.revision_run, args.strict_run
    samples = read_jsonl(run / "results/cost_benchmark/locked_timing_queries.jsonl")
    output = []
    for language in ("en", "zh"):
        cache = load_candidate_cache(strict, language)
        selected = [row for row in samples if row["language"] == language]
        wanted = {item["corpus_id"] for row in selected for item in cache[row["query_id"]]["monoqwen2vl_candidates"]}
        started = time.perf_counter()
        text, title, blocks = load_evidence(INDEX_DIR, language, wanted)
        sources = load_source_scores(ROUTER_DIR / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")
        load_seconds = time.perf_counter() - started
        for sample in selected:
            qid, outer = sample["query_id"], sample["outer_fold"]
            config = read_json(run / "results/fusion_ablations_deduplicated_v2/outer_configs" / f"{language}_outer_{outer}.json")
            state = read_json(strict / "tasks" / task_id("outer", language, outer) / "state.json")
            scores, _ = load_fin_scores(Path(state["prediction_path"]))
            for route in ("three_score", "complete_system"):
                original = config["e1_fixed_four_score"]
                score_key = "rerank_score" if route == "three_score" else "finconrank_score"
                items = deduplicate_candidates(direct_candidates(add_fincon(cache[qid]["monoqwen2vl_candidates"], scores[qid]), score_key))
                weights = config["metric_aware"]["weights"] if route == "three_score" else original["metric_aware"]["weights"]
                started = time.perf_counter()
                metric = apply_metric([qid], {qid: items}, score_key, weights, cache, text, title, blocks, sources, "cost_only")
                mapping = metric_score_map(metric)[qid]
                metric_seconds = time.perf_counter() - started
                started = time.perf_counter()
                if route == "three_score":
                    ranked = fuse_matched(items, mapping, config["normalizer_three"], np.asarray(config["e1_three_score"]["weights"]), False, "cost_only")
                else:
                    ranked = fuse(qid, items, mapping, original["normalizer"], np.asarray(original["fusion"]["weights_colq_mono_fincon_metric"]))
                fusion_seconds = time.perf_counter() - started
                output.append({"language": language, "query_id": qid, "route": route,
                               "candidate_count": len(ranked), "metric_feature_and_calibration_seconds": metric_seconds,
                               "normalization_fusion_sort_seconds": fusion_seconds,
                               "cpu_total_seconds": metric_seconds + fusion_seconds,
                               "shared_language_evidence_loading_seconds": load_seconds,
                               "boundary": "Warm cached evidence; exact frozen parameters; image scoring measured separately; no online combined wall-time claim"})
    atomic_csv(run / "results/cost_benchmark/cpu_actual_calibration.csv", output)


if __name__ == "__main__":
    main()
