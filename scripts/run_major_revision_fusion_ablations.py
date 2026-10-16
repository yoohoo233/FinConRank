#!/usr/bin/env python3
"""Strict revision-driven E1/E2 fusion ablations on frozen nested scores.

The command is deliberately split into lock/select/freeze/evaluate gates so that
outer-test qrels are not read before outer predictions are frozen and hashed.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_finconrank import feature_rows, load_evidence, load_source_scores, metric_vector, rank_from_items
from evaluate_finconrank_strict_nested import apply_metric, clustered_test, holm, load_fin_scores
from finconrank.common import METRICS, evaluate
from finconrank.strict_nested import (
    INDEX_DIR,
    LANGUAGES,
    OUTER_FOLDS,
    ROUTER_DIR,
    SEED,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    load_candidate_cache,
    qrels_path,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    task_id,
    utc_now,
)
from run_financial_metric_aware_reranking import candidate_weights
from select_finconrank_strict_nested import add_fincon, assemble_inner_scores, direct_candidates, fuse


SYSTEMS = ("e1_three_score", "e1_fixed_four_score", "e2_zero_allowed_four_score")
ANALYSIS_ID = "fusion_ablations_deduplicated_v2"
PROTOCOL_STEM = "fusion_ablations_deduplicated_v2_protocol_lock"


def deduplicate_candidates(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first ranked occurrence of each corpus ID, matching corrected evaluation."""
    output = []
    seen = set()
    for item in values:
        cid = item["corpus_id"]
        if cid in seen:
            continue
        seen.add(cid)
        output.append(item)
    for rank, item in enumerate(output, 1):
        item["rank"] = rank
    return output


def result_dir(revision_run: Path) -> Path:
    return revision_run / "results" / ANALYSIS_ID


def protocol_path(revision_run: Path) -> Path:
    return revision_run / "protocol" / f"{PROTOCOL_STEM}.json"


def select_key(metrics: dict[str, float], index: int) -> tuple[float, float, float, int]:
    return metrics["nDCG@5"], metrics["nDCG@10"], metrics["MRR@10"], -index


def aggregate(
    qids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    values = {metric: [] for metric in METRICS}
    for qid in qids:
        metrics = evaluate(candidates[qid], qrels[qid])
        for metric in METRICS:
            values[metric].append(metrics[metric])
    return {metric: float(np.mean(items)) if items else 0.0 for metric, items in values.items()}


def fit_stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": 0.0, "std": 1.0, "count": 0}
    std = float(array.std())
    return {"mean": float(array.mean()), "std": std if std > 1e-12 else 1.0, "count": len(array)}


def zsigmoid(value: float, params: dict[str, float]) -> float:
    z = (value - params["mean"]) / params["std"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit_matched_normalizer(
    qids: list[str],
    mono: dict[str, list[dict[str, Any]]],
    metric_scores: dict[str, dict[str, float]],
    include_fincon: bool,
) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for qid in qids:
        for item in mono[qid]:
            values["colq"].append(float(item.get("score", 0.0)))
            values["mono"].append(float(item.get("rerank_score", 0.0)))
            if include_fincon:
                values["fincon"].append(float(item.get("finconrank_score", 0.0)))
        values["metric"].extend(metric_scores[qid].values())
    return {name: fit_stats(items) for name, items in values.items()}


def simplex_grid(dimensions: int, require_fincon_positive: bool = False) -> list[np.ndarray]:
    output = []
    for parts in itertools.product(range(0, 11, 2), repeat=dimensions):
        if sum(parts) != 10:
            continue
        if require_fincon_positive and dimensions == 4 and parts[2] <= 0:
            continue
        output.append(np.asarray(parts, dtype=float) / 10.0)
    return output


def fuse_matched(
    candidates: list[dict[str, Any]],
    metric_scores: dict[str, float],
    normalizer: dict[str, dict[str, float]],
    weights: np.ndarray,
    include_fincon: bool,
    run_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for original_rank, source in enumerate(candidates, 1):
        item = dict(source)
        components = [
            zsigmoid(float(item.get("score", 0.0)), normalizer["colq"]),
            zsigmoid(float(item.get("rerank_score", 0.0)), normalizer["mono"]),
        ]
        if include_fincon:
            components.append(zsigmoid(float(item.get("finconrank_score", 0.0)), normalizer["fincon"]))
        components.append(zsigmoid(float(metric_scores[item["corpus_id"]]), normalizer["metric"]))
        vector = np.asarray(components, dtype=float)
        item["fusion_components"] = vector.tolist()
        item["fusion_score"] = float(weights @ vector)
        item["pre_fusion_rank"] = original_rank
        item["run_name"] = run_name
        rows.append(item)
    rows.sort(key=lambda item: (item["fusion_score"], -item["pre_fusion_rank"]), reverse=True)
    for rank, item in enumerate(rows, 1):
        item["rank"] = rank
    return rows


def select_metric_weights(
    qids: list[str],
    feature_items: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    grid = candidate_weights(12, seed)
    options = []
    for index, weight in enumerate(grid):
        ranked = {
            qid: rank_from_items(
                feature_items[qid],
                [float(weight @ metric_vector(item)) for item in feature_items[qid]],
                "revision_inner_mono_metric_selection",
            )
            for qid in qids
        }
        metrics = aggregate(qids, ranked, qrels)
        options.append((select_key(metrics, index), index, weight, metrics))
    _, index, selected, metrics = max(options, key=lambda item: item[0])
    ranked = {
        qid: rank_from_items(
            feature_items[qid],
            [float(selected @ metric_vector(item)) for item in feature_items[qid]],
            "revision_inner_mono_metric_selected",
        )
        for qid in qids
    }
    return selected, {
        "weight_index": index,
        "weights": selected.tolist(),
        "candidate_count": len(grid),
        "selection_metrics": metrics,
    }, ranked


def metric_score_map(ranked: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    return {
        qid: {item["corpus_id"]: float(item["metric_aware_score"]) for item in items}
        for qid, items in ranked.items()
    }


def select_fusion(
    qids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    metric_scores: dict[str, dict[str, float]],
    normalizer: dict[str, dict[str, float]],
    grid: list[np.ndarray],
    qrels: dict[str, dict[str, int]],
    include_fincon: bool,
    run_name: str,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, int], int, np.ndarray, dict[str, float]] | None = None
    for index, weights in enumerate(grid):
        rankings = {
            qid: fuse_matched(
                candidates[qid], metric_scores[qid], normalizer, weights, include_fincon, run_name
            )
            for qid in qids
        }
        metrics = aggregate(qids, rankings, qrels)
        trace.append({"weight_index": index, "weights": weights.tolist(), **metrics})
        option = (select_key(metrics, index), index, weights, metrics)
        if best is None or option[0] > best[0]:
            best = option
    assert best is not None
    _, index, weights, metrics = best
    return weights, {
        "weight_index": index,
        "weights": weights.tolist(),
        "candidate_count": len(grid),
        "selection_metrics": metrics,
    }, trace


def protocol_material(strict_run: Path, revision_run: Path) -> dict[str, Any]:
    files = [
        strict_run / "protocol_lock.json",
        strict_run / "task_manifest.json",
        revision_run / "stage0" / "audit_summary.json",
        Path(__file__).resolve(),
        revision_run / "results" / "fusion_ablations" / "original_fixed_four_reconstruction_audit.json",
        revision_run / "protocol" / "locked_scripts" / "run_major_revision_fusion_ablations_v1.py",
    ]
    files.extend(strict_run / "outer_configs" / f"{language}_outer_{outer}.json" for language in LANGUAGES for outer in range(5))
    files.extend(strict_run / "inner_predictions" / f"inner_{language}_outer{outer}_inner{inner}.jsonl" for language in LANGUAGES for outer in range(5) for inner in range(4))
    files.extend(strict_run / "outer_predictions" / f"outer_{language}_fold{outer}.jsonl" for language in LANGUAGES for outer in range(5))
    return {
        "protocol_name": "FinConRank major-revision E1/E2 strict nested fusion ablations, deduplicated v2",
        "status": "locked_before_supplementary_outer_prediction_generation",
        "locked_at": utc_now(),
        "revision_driven_not_original_preregistration": True,
        "seed": SEED,
        "strict_run": str(strict_run),
        "revision_run": str(revision_run),
        "query_counts": {"en": 538, "zh": 854},
        "report_group_counts": {"en": 105, "zh": 196},
        "outer_folds": 5,
        "inner_folds_per_outer": 4,
        "candidate_policy": {
            "identity_source": "frozen MonoQwen2-VL top-20 for every E1/E2 system",
            "depth": 20,
            "empty_candidate_queries_retained": True,
            "duplicate_handling": "deduplicate corpus_id by first ranked occurrence before all inner selection and outer scoring; matches the corrected strict evaluator",
        },
        "e1": {
            "three_score_components": ["ColQwen2", "MonoQwen2-VL", "Mono-derived Metric-Aware"],
            "metric_weight_candidates": 221,
            "fusion_grid_step": 0.2,
            "fusion_grid_size": len(simplex_grid(3)),
            "fixed_four_comparator": "original strict full_score_fusion reconstructed from each frozen outer config",
            "interpretation": "total incremental effect of the original FinConRank-including full fusion over the independently optimized no-FinCon three-score baseline",
        },
        "e2": {
            "four_score_components": ["ColQwen2", "MonoQwen2-VL", "FinConRank", "Mono-derived Metric-Aware"],
            "fusion_grid_step": 0.2,
            "fusion_grid_size": len(simplex_grid(4)),
            "finconrank_weight_constraint": "w_F >= 0",
            "contains_exact_e1_three_score_subspace": True,
            "shared_metric_component_with_e1": True,
        },
        "selection": {
            "scope": "current outer fold cross-fitted inner OOF only",
            "primary": "nDCG@5",
            "tie_break": ["nDCG@10", "MRR@10", "lowest deterministic grid index"],
            "normalization": "outer-development inner-OOF global mean/std followed by sigmoid",
            "outer_test_access": "predictions frozen and hashed before test qrels are opened",
        },
        "supplementary_family": {
            "size": 8,
            "comparisons": [
                "E1 fixed original four-score minus E1 three-score",
                "E2 zero-allowed four-score minus E1 three-score",
            ],
            "languages": ["en", "zh"],
            "metrics": ["nDCG@10", "MRR@10"],
            "cluster_unit": "source report group",
            "bootstrap": "10000 report-group percentile samples; CI not multiplicity-adjusted",
            "test": "10000 report-group two-sided sign flips",
            "multiple_comparison": "Holm over all 8 raw p-values",
            "seed_rule": "42 + comparison_index * 31",
            "comparison_index_map": [
                {
                    "comparison_index": index,
                    "comparison": comparison,
                    "language": language,
                    "metric": metric,
                    "seed": SEED + index * 31,
                }
                for index, (comparison, language, metric) in enumerate(
                    (comparison, language, metric)
                    for comparison in (
                        "E1_fixed_four_minus_three",
                        "E2_zero_allowed_four_minus_three",
                    )
                    for language in LANGUAGES
                    for metric in ("nDCG@10", "MRR@10")
                )
            ],
        },
        "prohibitions": {
            "historical_global_oof_scores": True,
            "human_audit_labels": True,
            "outer_test_qrels_for_selection": True,
            "filter_empty_candidate_queries": True,
        },
        "supersedes": {
            "analysis_id": "fusion_ablations",
            "reason": "v1 completion audit detected one query with duplicate corpus IDs, producing Recall and nDCG above 1 and preventing exact reconstruction of corrected historical rankings",
            "failed_audit": str(revision_run / "results" / "fusion_ablations" / "original_fixed_four_reconstruction_audit.json"),
            "v1_locked_script": str(revision_run / "protocol" / "locked_scripts" / "run_major_revision_fusion_ablations_v1.py"),
        },
        "input_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
    }


def command_lock(strict_run: Path, revision_run: Path) -> None:
    output = protocol_path(revision_run)
    if output.exists():
        raise RuntimeError(f"protocol already exists: {output}")
    protocol = protocol_material(strict_run, revision_run)
    protocol["protocol_sha256"] = stable_hash(protocol)
    atomic_json(output, protocol)
    (revision_run / "protocol" / f"{PROTOCOL_STEM}.sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="ascii"
    )
    print(json.dumps({"status": "locked", "path": str(output), "sha256": sha256_file(output)}, indent=2))


def verify_protocol(strict_run: Path, revision_run: Path) -> dict[str, Any]:
    path = protocol_path(revision_run)
    protocol = read_json(path)
    if protocol.get("status") != "locked_before_supplementary_outer_prediction_generation":
        raise RuntimeError("supplementary protocol is not locked")
    for entry in protocol["input_files"]:
        source = Path(entry["path"])
        if source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]:
            raise RuntimeError(f"locked input changed: {source}")
    if Path(protocol["strict_run"]) != strict_run or Path(protocol["revision_run"]) != revision_run:
        raise RuntimeError("run path mismatch")
    return protocol


def command_select(strict_run: Path, revision_run: Path) -> None:
    protocol = verify_protocol(strict_run, revision_run)
    output_dir = result_dir(revision_run)
    config_dir = output_dir / "outer_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    config_manifest = []
    for language in LANGUAGES:
        cache = load_candidate_cache(strict_run, language)
        for outer in range(OUTER_FOLDS):
            split = read_json(strict_run / "splits" / f"outer_{language}_fold{outer}.json")
            qids = split["development_qids"]
            qrels = read_json(qrels_path(strict_run, language, outer, "development"))
            fin_scores, provenance = assemble_inner_scores(strict_run, language, outer, set(qids))
            mono: dict[str, list[dict[str, Any]]] = {}
            for qid in qids:
                mono[qid] = deduplicate_candidates(direct_candidates(
                    add_fincon(cache[qid]["monoqwen2vl_candidates"], fin_scores[qid]), "rerank_score"
                ))
            wanted = {item["corpus_id"] for qid in qids for item in mono[qid]}
            page_text, page_title, blocks = load_evidence(INDEX_DIR, language, wanted)
            source_scores = load_source_scores(ROUTER_DIR / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")
            feature_items = {
                qid: feature_rows(
                    {"query_id": qid, "candidates": mono[qid]},
                    cache[qid]["constraints"], page_text, page_title, blocks,
                    source_scores.get(qid, {}), "rerank_score",
                )
                for qid in qids
            }
            metric_seed = SEED + outer * 100 + (0 if language == "en" else 10000)
            metric_weights, metric_config, metric_ranked = select_metric_weights(qids, feature_items, qrels, metric_seed)
            metric_scores = metric_score_map(metric_ranked)
            normalizer_three = fit_matched_normalizer(qids, mono, metric_scores, include_fincon=False)
            normalizer_four = fit_matched_normalizer(qids, mono, metric_scores, include_fincon=True)
            three_weights, three_config, three_trace = select_fusion(
                qids, mono, metric_scores, normalizer_three, simplex_grid(3), qrels,
                include_fincon=False, run_name="e1_three_score",
            )
            four_weights, four_config, four_trace = select_fusion(
                qids, mono, metric_scores, normalizer_four, simplex_grid(4), qrels,
                include_fincon=True, run_name="e2_zero_allowed_four_score",
            )
            original_config_path = strict_run / "outer_configs" / f"{language}_outer_{outer}.json"
            original_config = read_json(original_config_path)
            config = {
                "created_at": utc_now(),
                "language": language,
                "outer_fold": outer,
                "selection_scope": "strict current-outer cross-fitted inner OOF only",
                "inner_oof_query_count": len(qids),
                "candidate_policy": protocol["candidate_policy"],
                "metric_aware": {**metric_config, "seed": metric_seed},
                "normalizer_three": normalizer_three,
                "normalizer_four": normalizer_four,
                "e1_three_score": three_config,
                "e2_zero_allowed_four_score": four_config,
                "e2_finconrank_weight": float(four_weights[2]),
                "e2_selected_zero_finconrank_weight": bool(abs(float(four_weights[2])) < 1e-15),
                "e1_fixed_four_score": {
                    "source": str(original_config_path),
                    "source_sha256": sha256_file(original_config_path),
                    "source_config_sha256": original_config["config_sha256"],
                    "metric_aware": original_config["metric_aware"]["serial_metric_aware"],
                    "normalizer": original_config["normalizer"],
                    "fusion": original_config["fusion"],
                },
                "provenance_audit": provenance,
                "outer_test_qrels_read": False,
            }
            config["config_sha256"] = stable_hash(config)
            config_path = config_dir / f"{language}_outer_{outer}.json"
            atomic_json(config_path, config)
            config_manifest.append({
                "language": language,
                "outer_fold": outer,
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "config_sha256": config["config_sha256"],
                "e2_finconrank_weight": config["e2_finconrank_weight"],
            })
            for experiment, trace in (("E1_three_score", three_trace), ("E2_zero_allowed_four_score", four_trace)):
                for row in trace:
                    search_rows.append({"language": language, "outer_fold": outer, "experiment": experiment, **row})
            provenance_rows.append({
                "language": language,
                "outer_fold": outer,
                "expected_queries": provenance["expected_queries"],
                "observed_queries": provenance["observed_queries"],
                "missing": len(provenance["missing"]),
                "extra": len(provenance["extra"]),
                "errors": len(provenance["errors"]),
                "status": "PASS" if not provenance["missing"] and not provenance["extra"] and not provenance["errors"] else "FAIL",
            })
    atomic_csv(output_dir / "inner_search_trace.csv", search_rows)
    atomic_csv(output_dir / "inner_provenance_audit.csv", provenance_rows)
    atomic_json(output_dir / "outer_config_manifest.json", {"created_at": utc_now(), "configs": config_manifest})
    (output_dir / "OUTER_CONFIGS_LOCKED").write_text(
        json.dumps({"created_at": utc_now(), "manifest_sha256": sha256_file(output_dir / "outer_config_manifest.json")}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "selected", "configs": len(config_manifest), "search_rows": len(search_rows)}, indent=2))


def command_freeze(strict_run: Path, revision_run: Path) -> None:
    verify_protocol(strict_run, revision_run)
    output_dir = result_dir(revision_run)
    if not (output_dir / "OUTER_CONFIGS_LOCKED").is_file():
        raise RuntimeError("outer configs are not locked")
    prediction_dir = output_dir / "outer_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for language in LANGUAGES:
        cache = load_candidate_cache(strict_run, language)
        for outer in range(OUTER_FOLDS):
            split = read_json(strict_run / "splits" / f"outer_{language}_fold{outer}.json")
            qids = split["test_qids"]
            config_path = output_dir / "outer_configs" / f"{language}_outer_{outer}.json"
            config = read_json(config_path)
            state = read_json(strict_run / "tasks" / task_id("outer", language, outer) / "state.json")
            fin_scores, raw_score_rows = load_fin_scores(Path(state["prediction_path"]))
            if set(fin_scores) != set(qids):
                raise RuntimeError(f"outer score coverage mismatch: {language} outer {outer}")
            mono = {
                qid: deduplicate_candidates(direct_candidates(add_fincon(cache[qid]["monoqwen2vl_candidates"], fin_scores[qid]), "rerank_score"))
                for qid in qids
            }
            wanted = {item["corpus_id"] for qid in qids for item in mono[qid]}
            page_text, page_title, blocks = load_evidence(INDEX_DIR, language, wanted)
            source_scores = load_source_scores(ROUTER_DIR / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")
            mono_metric = apply_metric(
                qids, mono, "rerank_score", config["metric_aware"]["weights"],
                cache, page_text, page_title, blocks, source_scores, "revision_outer_mono_metric",
            )
            mono_metric_scores = metric_score_map(mono_metric)
            original = config["e1_fixed_four_score"]
            serial = {
                qid: deduplicate_candidates(direct_candidates(add_fincon(cache[qid]["monoqwen2vl_candidates"], fin_scores[qid]), "finconrank_score"))
                for qid in qids
            }
            original_metric = apply_metric(
                qids, serial, "finconrank_score", original["metric_aware"]["weights"],
                cache, page_text, page_title, blocks, source_scores, "revision_reconstructed_original_serial_metric",
            )
            original_metric_scores = metric_score_map(original_metric)
            rows = []
            for qid in qids:
                three = fuse_matched(
                    mono[qid], mono_metric_scores[qid], config["normalizer_three"],
                    np.asarray(config["e1_three_score"]["weights"], dtype=float), False, "e1_three_score",
                )
                zero = fuse_matched(
                    mono[qid], mono_metric_scores[qid], config["normalizer_four"],
                    np.asarray(config["e2_zero_allowed_four_score"]["weights"], dtype=float), True,
                    "e2_zero_allowed_four_score",
                )
                fixed = fuse(
                    qid, serial[qid], original_metric_scores[qid], original["normalizer"],
                    np.asarray(original["fusion"]["weights_colq_mono_fincon_metric"], dtype=float),
                )
                for item in fixed:
                    item["run_name"] = "e1_fixed_four_score"
                systems = {"e1_three_score": three, "e1_fixed_four_score": fixed, "e2_zero_allowed_four_score": zero}
                rows.append({
                    "query_id": qid,
                    "language": language,
                    "outer_fold": outer,
                    "source_report_group": cache[qid]["group_id"],
                    "query": cache[qid]["query"],
                    "query_type": cache[qid]["query_type"],
                    "outer_config_path": str(config_path),
                    "outer_config_sha256": sha256_file(config_path),
                    "finconrank_checkpoint_sha256": state["checkpoint_sha256"],
                    "raw_finconrank_prediction_sha256": state["inference"]["prediction_sha256"],
                    "systems": systems,
                    "qrels_included": False,
                })
            output = prediction_dir / f"{language}_outer_{outer}.jsonl"
            atomic_jsonl(output, rows)
            manifest.append({
                "language": language, "outer_fold": outer, "query_count": len(rows),
                "path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output),
                "contains_qrels": False,
            })
    freeze = {
        "status": "frozen_before_outer_test_qrels_evaluation",
        "created_at": utc_now(),
        "predictions": manifest,
        "query_count": sum(item["query_count"] for item in manifest),
        "outer_test_metrics_seen_by_this_script": False,
    }
    atomic_json(output_dir / "outer_prediction_freeze_manifest.json", freeze)
    (output_dir / "OUTER_PREDICTIONS_FROZEN").write_text(
        f"{sha256_file(output_dir / 'outer_prediction_freeze_manifest.json')}  outer_prediction_freeze_manifest.json\n",
        encoding="ascii",
    )
    print(json.dumps({"status": "frozen", "files": len(manifest), "queries": freeze["query_count"]}, indent=2))


def verify_freeze(result_dir: Path) -> dict[str, Any]:
    freeze = read_json(result_dir / "outer_prediction_freeze_manifest.json")
    if freeze.get("status") != "frozen_before_outer_test_qrels_evaluation" or len(freeze.get("predictions", [])) != 10:
        raise RuntimeError("outer prediction freeze is incomplete")
    for entry in freeze["predictions"]:
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"frozen prediction changed: {path}")
    if (result_dir / "OUTER_TEST_EVALUATION_COMPLETE").exists():
        raise RuntimeError("outer test was already evaluated")
    return freeze


def command_evaluate(strict_run: Path, revision_run: Path) -> None:
    verify_protocol(strict_run, revision_run)
    output_dir = result_dir(revision_run)
    freeze = verify_freeze(output_dir)
    query_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for entry in freeze["predictions"]:
        language = entry["language"]
        outer = int(entry["outer_fold"])
        qrels = read_json(qrels_path(strict_run, language, outer, "test"))
        rows = read_jsonl(Path(entry["path"]))
        for row in rows:
            qid = row["query_id"]
            systems = {
                system: {
                    "metrics": evaluate(row["systems"][system], qrels[qid]),
                    "candidates": row["systems"][system],
                }
                for system in SYSTEMS
            }
            query_rows.append({
                "query_id": qid, "language": language, "outer_fold": outer,
                "source_report_group": row["source_report_group"], "query_type": row["query_type"],
                "qrels": qrels[qid],
                "systems": systems,
            })
        for system in SYSTEMS:
            subset = [row for row in query_rows if row["language"] == language and row["outer_fold"] == outer]
            fold_rows.append({
                "language": language, "outer_fold": outer, "run": system, "query_count": len(subset),
                **{metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in subset])) for metric in METRICS},
            })
    expected = {"en": 538, "zh": 854}
    for language, count in expected.items():
        subset = [row for row in query_rows if row["language"] == language]
        if len(subset) != count or len({row["query_id"] for row in subset}) != count:
            raise RuntimeError(f"query coverage mismatch for {language}: {len(subset)}")
    atomic_jsonl(output_dir / "query_level_evaluated_predictions.jsonl", query_rows)
    metrics_rows = []
    for language in LANGUAGES:
        subset = [row for row in query_rows if row["language"] == language]
        for system in SYSTEMS:
            metrics_rows.append({
                "language": language, "run": system, "query_count": len(subset),
                **{metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in subset])) for metric in METRICS},
            })
    atomic_csv(output_dir / "metrics_summary.csv", metrics_rows)
    atomic_csv(output_dir / "fold_metrics.csv", fold_rows)
    delta_rows = []
    for language in LANGUAGES:
        base = next(row for row in metrics_rows if row["language"] == language and row["run"] == "e1_three_score")
        for system in ("e1_fixed_four_score", "e2_zero_allowed_four_score"):
            value = next(row for row in metrics_rows if row["language"] == language and row["run"] == system)
            delta_rows.append({
                "language": language, "run": system, "baseline": "e1_three_score",
                **{f"delta_{metric}": value[metric] - base[metric] for metric in METRICS},
            })
    atomic_csv(output_dir / "delta_vs_three_score.csv", delta_rows)
    significance = []
    comparison_index = 0
    for comparison_id, system in (
        ("E1_fixed_four_minus_three", "e1_fixed_four_score"),
        ("E2_zero_allowed_four_minus_three", "e2_zero_allowed_four_score"),
    ):
        for language in LANGUAGES:
            subset = [row for row in query_rows if row["language"] == language]
            for metric in ("nDCG@10", "MRR@10"):
                deltas = [
                    row["systems"][system]["metrics"][metric] - row["systems"]["e1_three_score"]["metrics"][metric]
                    for row in subset
                ]
                seed = SEED + comparison_index * 31
                significance.append({
                    "family": "revision_e1_e2_eight_tests",
                    "comparison_index": comparison_index,
                    "comparison": comparison_id,
                    "system": system,
                    "baseline": "e1_three_score",
                    "language": language,
                    "metric": metric,
                    "seed": seed,
                    **clustered_test(deltas, [row["source_report_group"] for row in subset], seed, samples=10000),
                })
                comparison_index += 1
    holm(significance)
    atomic_csv(output_dir / "statistical_significance.csv", significance)
    atomic_json(output_dir / "statistical_significance.json", {"tests": significance})

    historical_path = strict_run / "strict_nested_outer_query_predictions.jsonl"
    historical = {
        (row["language"], int(row["outer_fold"]), row["query_id"]): [item["corpus_id"] for item in row["systems"]["full_score_fusion"]["candidates"]]
        for row in read_jsonl(historical_path)
    }
    rank_mismatches = []
    for row in query_rows:
        key = (row["language"], int(row["outer_fold"]), row["query_id"])
        reconstructed = [item["corpus_id"] for item in row["systems"]["e1_fixed_four_score"]["candidates"]]
        if reconstructed != historical.get(key):
            rank_mismatches.append({"language": key[0], "outer_fold": key[1], "query_id": key[2]})
    reconstruction_audit = {
        "historical_prediction_path": str(historical_path),
        "historical_prediction_sha256": sha256_file(historical_path),
        "compared_queries": len(query_rows),
        "rank_mismatch_count": len(rank_mismatches),
        "rank_mismatches": rank_mismatches,
        "passed": not rank_mismatches,
    }
    atomic_json(output_dir / "original_fixed_four_reconstruction_audit.json", reconstruction_audit)
    if rank_mismatches:
        raise RuntimeError(f"original fixed-four reconstruction mismatch: {len(rank_mismatches)} queries")

    reloaded = read_jsonl(output_dir / "query_level_evaluated_predictions.jsonl")
    max_error = 0.0
    for expected_row in metrics_rows:
        subset = [row for row in reloaded if row["language"] == expected_row["language"]]
        for metric in METRICS:
            actual = float(np.mean([
                evaluate(row["systems"][expected_row["run"]]["candidates"], row["qrels"])[metric]
                for row in subset
            ]))
            max_error = max(max_error, abs(actual - expected_row[metric]))
    audit = {
        "created_at": utc_now(),
        "frozen_prediction_manifest_sha256": sha256_file(output_dir / "outer_prediction_freeze_manifest.json"),
        "evaluated_query_prediction_sha256": sha256_file(output_dir / "query_level_evaluated_predictions.jsonl"),
        "maximum_absolute_aggregate_error": max_error,
        "tolerance": 1e-12,
        "reconstruction_audit_passed": reconstruction_audit["passed"],
        "query_counts": expected,
        "passed": max_error < 1e-12 and reconstruction_audit["passed"],
    }
    atomic_json(output_dir / "evaluation_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"evaluation audit failed: {audit}")
    (output_dir / "OUTER_TEST_EVALUATION_COMPLETE").write_text(
        json.dumps({"completed_at": utc_now(), "audit_sha256": sha256_file(output_dir / "evaluation_audit.json")}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "evaluated", "queries": len(query_rows), "tests": len(significance)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock", "select", "freeze", "evaluate"))
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    strict_run = args.strict_run.resolve()
    revision_run = args.revision_run.resolve()
    if args.command == "lock":
        command_lock(strict_run, revision_run)
    elif args.command == "select":
        command_select(strict_run, revision_run)
    elif args.command == "freeze":
        command_freeze(strict_run, revision_run)
    else:
        command_evaluate(strict_run, revision_run)


if __name__ == "__main__":
    main()
