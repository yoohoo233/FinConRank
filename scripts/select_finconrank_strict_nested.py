#!/usr/bin/env python3
"""Select each outer-fold system using only strict cross-fitted inner scores."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_finconrank import feature_rows, load_evidence, load_source_scores, metric_vector, rank_from_items
from finconrank.common import METRICS, evaluate
from finconrank.strict_nested import (
    CANDIDATE_DEPTH,
    INDEX_DIR,
    LANGUAGES,
    OUTER_FOLDS,
    ROUTER_DIR,
    SEED,
    atomic_csv,
    atomic_json,
    load_candidate_cache,
    qrels_path,
    read_json,
    read_jsonl,
    stable_hash,
    task_id,
    utc_now,
)
from run_financial_metric_aware_reranking import candidate_weights


SELECTABLE_ORDER = [
    "full_score_fusion", "serial_metric_aware", "replacement_metric_aware",
    "serial", "replacement", "mono_metric_aware",
]


def direct_candidates(values: list[dict[str, Any]], field: str | None = None) -> list[dict[str, Any]]:
    rows = [dict(item) for item in values[:CANDIDATE_DEPTH]]
    if field:
        rows.sort(
            key=lambda item: (
                float(item.get(field, -1e9) if item.get(field) is not None else -1e9),
                -int(item.get("rank", 999)),
            ),
            reverse=True,
        )
    for rank, item in enumerate(rows, 1):
        item["rank"] = rank
    return rows


def add_fincon(values: list[dict[str, Any]], scores: dict[str, float]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in values[:CANDIDATE_DEPTH]]
    for item in rows:
        item["finconrank_score"] = scores[item["corpus_id"]]
    return rows


def aggregate(qids: list[str], candidates: dict[str, list[dict[str, Any]]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    values = {metric: [] for metric in METRICS}
    for qid in qids:
        row = evaluate(candidates[qid], qrels[qid])
        for metric in METRICS:
            values[metric].append(row[metric])
    return {metric: float(np.mean(items)) if items else 0.0 for metric, items in values.items()}


def select_metric_weights(
    qids: list[str],
    items: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    grid = candidate_weights(12, seed)
    ranked_options = []
    for index, weight in enumerate(grid):
        candidate_map = {
            qid: rank_from_items(
                items[qid], [float(weight @ metric_vector(item)) for item in items[qid]],
                "strict_inner_metric_selection",
            )
            for qid in qids
        }
        metrics = aggregate(qids, candidate_map, qrels)
        ranked_options.append((metrics["nDCG@5"], metrics["nDCG@10"], metrics["MRR@10"], -index, index, weight))
    _, _, _, _, index, selected = max(ranked_options)
    output = {
        qid: rank_from_items(
            items[qid], [float(selected @ metric_vector(item)) for item in items[qid]],
            "strict_inner_metric_selected",
        )
        for qid in qids
    }
    return selected, {
        "weight_index": index, "weights": selected.tolist(), "candidate_count": len(grid),
        "selection_metrics": aggregate(qids, output, qrels),
    }, output


def fit_normalizer(
    qids: list[str],
    serial: dict[str, list[dict[str, Any]]],
    metric_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for qid in qids:
        for item in serial[qid]:
            values["colq"].append(float(item.get("score", 0.0)))
            values["mono"].append(float(item.get("rerank_score", 0.0)))
            values["fincon"].append(float(item.get("finconrank_score", 0.0)))
        values["metric"].extend(metric_scores[qid].values())
    result = {}
    for name, items in values.items():
        arr = np.asarray(items, dtype=float)
        std = float(arr.std())
        result[name] = {"mean": float(arr.mean()), "std": std if std > 1e-12 else 1.0, "count": len(items)}
    return result


def zsigmoid(value: float, params: dict[str, float]) -> float:
    z = (value - params["mean"]) / params["std"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fusion_grid() -> list[np.ndarray]:
    return [
        np.asarray(parts, dtype=float) / 10.0
        for parts in itertools.product(range(0, 11, 2), repeat=4)
        if sum(parts) == 10 and parts[2] > 0
    ]


def fuse(
    qid: str,
    serial: list[dict[str, Any]],
    metric_scores: dict[str, float],
    normalizer: dict[str, dict[str, float]],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for original_rank, source in enumerate(serial, 1):
        item = dict(source)
        vector = np.asarray([
            zsigmoid(float(item.get("score", 0.0)), normalizer["colq"]),
            zsigmoid(float(item.get("rerank_score", 0.0)), normalizer["mono"]),
            zsigmoid(float(item.get("finconrank_score", 0.0)), normalizer["fincon"]),
            zsigmoid(float(metric_scores[item["corpus_id"]]), normalizer["metric"]),
        ])
        item["fusion_components"] = vector.tolist()
        item["fusion_score"] = float(weights @ vector)
        item["pre_fusion_rank"] = original_rank
        rows.append(item)
    rows.sort(key=lambda item: (item["fusion_score"], -item["pre_fusion_rank"]), reverse=True)
    for rank, item in enumerate(rows, 1):
        item["rank"] = rank
    return rows


def assemble_inner_scores(run_dir: Path, language: str, outer: int, expected_qids: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    scores: dict[str, dict[str, float]] = {}
    audit = {"language": language, "outer_fold": outer, "inner_units": [], "errors": []}
    split_outer = read_json(run_dir / f"splits/outer_{language}_fold{outer}.json")
    outer_test_groups = set(split_outer["test_report_groups"])
    for inner in range(4):
        identifier = task_id("inner", language, outer, inner)
        split = read_json(run_dir / f"splits/{identifier}.json")
        state = read_json(run_dir / f"tasks/{identifier}/state.json")
        if state.get("status") != "complete":
            raise RuntimeError(f"inner task incomplete: {identifier}")
        rows = read_jsonl(Path(state["prediction_path"]))
        train_groups = set(split["train_report_groups"])
        validation_groups = set(split["validation_report_groups"])
        if train_groups & validation_groups or train_groups & outer_test_groups:
            raise RuntimeError(f"group overlap in {identifier}")
        observed = set()
        for row in rows:
            qid = row["query_id"]
            if row["source_report_group"] not in validation_groups:
                audit["errors"].append({"task_id": identifier, "query_id": qid, "reason": "not in inner validation group"})
            if row["checkpoint_training_report_groups_sha256"] != split["train_groups_sha256"]:
                audit["errors"].append({"task_id": identifier, "query_id": qid, "reason": "training group hash mismatch"})
            if qid in scores:
                audit["errors"].append({"task_id": identifier, "query_id": qid, "reason": "duplicate cross-fitted query"})
            observed.add(qid)
            scores[qid] = {
                candidate["candidate_id"]: float(candidate["finconrank_score"])
                for candidate in row["candidates"] if candidate.get("finconrank_score") is not None
            }
        audit["inner_units"].append({
            "task_id": identifier, "expected_queries": len(split["validation_qids"]),
            "observed_queries": len(observed), "checkpoint_sha256": state["checkpoint_sha256"],
            "physical_gpu_id": state["physical_gpu_id"],
        })
    missing = sorted(expected_qids - set(scores))
    extra = sorted(set(scores) - expected_qids)
    audit.update({"expected_queries": len(expected_qids), "observed_queries": len(scores), "missing": missing, "extra": extra})
    if missing or extra or audit["errors"]:
        raise RuntimeError(f"strict inner OOF assembly failed: {audit}")
    return scores, audit


def select_one(run_dir: Path, language: str, outer: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cache = load_candidate_cache(run_dir, language)
    outer_split = read_json(run_dir / f"splits/outer_{language}_fold{outer}.json")
    qids = outer_split["development_qids"]
    qrels = read_json(qrels_path(run_dir, language, outer, "development"))
    fin_scores, provenance = assemble_inner_scores(run_dir, language, outer, set(qids))
    direct: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for qid in qids:
        source = cache[qid]
        direct["colqwen2"][qid] = direct_candidates(source["colqwen2_candidates"])
        direct["monoqwen2vl"][qid] = direct_candidates(source["monoqwen2vl_candidates"], "rerank_score")
        direct["replacement"][qid] = direct_candidates(
            add_fincon(source["colqwen2_candidates"], fin_scores[qid]), "finconrank_score"
        )
        direct["serial"][qid] = direct_candidates(
            add_fincon(source["monoqwen2vl_candidates"], fin_scores[qid]), "finconrank_score"
        )
    wanted = {item["corpus_id"] for mode in ("monoqwen2vl", "replacement", "serial") for qid in qids for item in direct[mode][qid]}
    page_text, page_title, blocks = load_evidence(INDEX_DIR, language, wanted)
    source_scores = load_source_scores(ROUTER_DIR / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")
    metric_sources = {
        "mono_metric_aware": ("monoqwen2vl", "rerank_score"),
        "replacement_metric_aware": ("replacement", "finconrank_score"),
        "serial_metric_aware": ("serial", "finconrank_score"),
    }
    metric_params, metric_scores = {}, {}
    for output_mode, (input_mode, score_field) in metric_sources.items():
        items = {}
        for qid in qids:
            row = {"query_id": qid, "candidates": direct[input_mode][qid]}
            items[qid] = feature_rows(
                row, cache[qid]["constraints"], page_text, page_title, blocks,
                source_scores.get(qid, {}), score_field,
            )
        weights, params, ranked = select_metric_weights(
            qids, items, qrels,
            SEED + outer * 100 + (0 if language == "en" else 10000) + len(metric_params),
        )
        metric_params[output_mode] = params
        direct[output_mode] = ranked
        metric_scores[output_mode] = {
            qid: {item["corpus_id"]: float(item["metric_aware_score"]) for item in ranked[qid]}
            for qid in qids
        }
    normalizer = fit_normalizer(qids, direct["serial"], metric_scores["serial_metric_aware"])
    fusion_options = []
    for index, weights in enumerate(fusion_grid()):
        candidates = {
            qid: fuse(qid, direct["serial"][qid], metric_scores["serial_metric_aware"][qid], normalizer, weights)
            for qid in qids
        }
        metrics = aggregate(qids, candidates, qrels)
        fusion_options.append((metrics["nDCG@5"], metrics["nDCG@10"], metrics["MRR@10"], -index, index, weights))
    _, _, _, _, fusion_index, fusion_weights = max(fusion_options)
    direct["full_score_fusion"] = {
        qid: fuse(qid, direct["serial"][qid], metric_scores["serial_metric_aware"][qid], normalizer, fusion_weights)
        for qid in qids
    }
    mode_metrics = {mode: aggregate(qids, candidates, qrels) for mode, candidates in direct.items()}
    order_index = {mode: index for index, mode in enumerate(SELECTABLE_ORDER)}
    selected_mode = max(
        SELECTABLE_ORDER,
        key=lambda mode: (
            mode_metrics[mode]["nDCG@5"], mode_metrics[mode]["nDCG@10"],
            mode_metrics[mode]["MRR@10"], -order_index[mode],
        ),
    )
    config = {
        "created_at": utc_now(), "language": language, "outer_fold": outer,
        "selection_scope": "strict current-outer cross-fitted inner OOF only",
        "inner_oof_query_count": len(qids), "candidate_depth": CANDIDATE_DEPTH,
        "model_epochs": 3, "model_checkpoint_rule": "fixed epoch 3",
        "metric_aware": metric_params, "normalizer": normalizer,
        "fusion": {
            "weight_index": fusion_index,
            "weights_colq_mono_fincon_metric": fusion_weights.tolist(),
            "candidate_count": len(fusion_grid()),
            "selection_metrics": mode_metrics["full_score_fusion"],
        },
        "deployment_metrics": {mode: mode_metrics[mode] for mode in SELECTABLE_ORDER},
        "selected_deployment": selected_mode,
        "selected_deployment_metrics": mode_metrics[selected_mode],
        "outer_test_metrics_seen": False,
        "provenance_audit": provenance,
    }
    config["config_sha256"] = stable_hash(config)
    rows = [
        {"language": language, "outer_fold": outer, "run": mode, **metrics}
        for mode, metrics in mode_metrics.items()
    ]
    return config, rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    all_rows, all_audits, configs = [], [], []
    for language in LANGUAGES:
        for outer in range(OUTER_FOLDS):
            config, rows, audit = select_one(run_dir, language, outer)
            path = run_dir / f"outer_configs/{language}_outer_{outer}.json"
            atomic_json(path, config)
            all_rows.extend(rows)
            all_audits.append(audit)
            configs.append({
                "language": language, "outer_fold": outer,
                "config_path": str(path.resolve()), "config_sha256": config["config_sha256"],
                "selected_deployment": config["selected_deployment"],
            })
    atomic_csv(run_dir / "inner_oof_selection_metrics.csv", all_rows)
    atomic_json(run_dir / "inner_oof_provenance_audit.json", {
        "created_at": utc_now(), "passed": all(not audit["errors"] and not audit["missing"] and not audit["extra"] for audit in all_audits),
        "folds": all_audits,
    })
    manifest = read_json(run_dir / "task_manifest.json")
    if any(task["status"] != "complete" for task in manifest["tasks"] if task["kind"] == "inner"):
        raise RuntimeError("cannot create outer tasks before all inner tasks complete")
    existing = {task["task_id"] for task in manifest["tasks"]}
    for language in LANGUAGES:
        for outer in range(OUTER_FOLDS):
            identifier = task_id("outer", language, outer)
            if identifier not in existing:
                manifest["tasks"].append({
                    "task_id": identifier, "kind": "outer", "language": language,
                    "outer_fold": outer, "inner_fold": None, "status": "pending",
                    "attempts": 0, "micro_batch_size": 2, "gradient_accumulation": 2,
                    "config_path": str((run_dir / f"outer_configs/{language}_outer_{outer}.json").resolve()),
                })
    manifest["status"] = "outer_pending"
    manifest["outer_configs"] = configs
    atomic_json(run_dir / "task_manifest.json", manifest)
    print(json.dumps({"status": "outer_configs_locked", "configs": configs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
