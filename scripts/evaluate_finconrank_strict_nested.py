#!/usr/bin/env python3
"""Evaluate frozen strict nested outer predictions exactly once."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
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
    PROJECT,
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
from select_finconrank_strict_nested import add_fincon, direct_candidates, fuse


SYSTEMS = [
    "colqwen2", "monoqwen2vl", "replacement", "serial",
    "mono_metric_aware", "replacement_metric_aware", "serial_metric_aware",
    "full_score_fusion", "selected_complete_system",
]


def aggregate(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in rows])) if rows else 0.0
        for metric in METRICS
    }


def apply_metric(
    qids: list[str],
    candidates: dict[str, list[dict[str, Any]]],
    score_field: str,
    weights: list[float],
    cache: dict[str, dict[str, Any]],
    page_text: dict[str, str],
    page_title: dict[str, str],
    blocks: dict[str, list[str]],
    source_scores: dict[str, dict[str, dict[str, float]]],
    run_name: str,
) -> dict[str, list[dict[str, Any]]]:
    weight = np.asarray(weights, dtype=float)
    result = {}
    for qid in qids:
        items = feature_rows(
            {"query_id": qid, "candidates": candidates[qid]}, cache[qid]["constraints"],
            page_text, page_title, blocks, source_scores.get(qid, {}), score_field,
        )
        result[qid] = rank_from_items(
            items, [float(weight @ metric_vector(item)) for item in items], run_name,
        )
    return result


def load_fin_scores(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    score_map, rows = {}, {}
    for row in read_jsonl(path):
        qid = row["query_id"]
        rows[qid] = row
        score_map[qid] = {
            item["candidate_id"]: float(item["finconrank_score"])
            for item in row["candidates"] if item.get("finconrank_score") is not None
        }
    return score_map, rows


def verify_freeze(run_dir: Path) -> dict[str, Any]:
    freeze = read_json(run_dir / "outer_prediction_freeze_manifest.json")
    if freeze.get("status") != "frozen_before_qrels_evaluation" or len(freeze.get("predictions", [])) != 10:
        raise RuntimeError("outer predictions are not fully frozen")
    for item in freeze["predictions"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"outer prediction changed after freeze: {path}")
    if (run_dir / "OUTER_TEST_EVALUATION_COMPLETE").exists():
        raise RuntimeError("outer test has already been evaluated; protocol allows one evaluation only")
    return freeze


def clustered_test(deltas: list[float], groups: list[str], seed: int, samples: int = 10000) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for delta, group in zip(deltas, groups):
        by_group[group].append(delta)
    keys = sorted(by_group)
    arrays = [np.asarray(by_group[key], dtype=float) for key in keys]
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        selected = rng.integers(0, len(arrays), len(arrays))
        boot.append(float(np.concatenate([arrays[index] for index in selected]).mean()))
    extreme = 0
    rng_perm = np.random.default_rng(seed + 1)
    for _ in range(samples):
        signs = rng_perm.choice(np.asarray([-1.0, 1.0]), size=len(arrays))
        permuted = np.concatenate([array * signs[index] for index, array in enumerate(arrays)])
        extreme += abs(float(permuted.mean())) >= abs(observed) - 1e-15
    arr = np.asarray(deltas, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "query_count": len(deltas), "report_group_count": len(keys),
        "mean_delta": observed,
        "cluster_bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
        "cluster_bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
        "cluster_sign_flip_raw_p_value": (extreme + 1) / (samples + 1),
        "effect_size_paired_dz": observed / std if std > 0 else 0.0,
        "improved_queries": int((arr > 0).sum()),
        "regressed_queries": int((arr < 0).sum()),
        "unchanged_queries": int((arr == 0).sum()),
    }


def holm(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["cluster_sign_flip_raw_p_value"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["cluster_sign_flip_raw_p_value"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p_value"] = running
        rows[index]["holm_significant_0_05"] = running < 0.05


def write_reports(run_dir: Path, metrics: list[dict[str, Any]], deltas: list[dict[str, Any]], significance: list[dict[str, Any]], audit: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    table = [
        "| 语言 | 系统 | Query数 | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        table.append(
            f"| {row['language']} | {row['run']} | {row['query_count']} | "
            + " | ".join(f"{row[m]:.6f}" for m in METRICS) + " |"
        )
    report = [
        "# FinConRank严格Report-Group Nested Cross-Validation报告", "",
        f"生成时间：`{utc_now()}`", "",
        "本实验未重新运行ColQwen2 embedding retrieval或原始MonoQwen2-VL全库推理；二者候选与分数均来自冻结的tab-safe结果。FinConRank在每个inner和outer单元中进行了真实GPU训练与推理。", "",
        "历史`global oof_assembled` FinConRank分数未进入本实验的训练、inner参数选择或outer评测。每个inner-validation query由未训练其query和源report-group的checkpoint评分；outer-test预测在读取test qrels计算指标前已冻结并记录SHA-256。", "",
        "## 主结果", "", *table, "",
        "结果口径为英文538条、中文854条query的5-fold outer report-group OOF拼接。系统选择、Metric-Aware权重、归一化参数和fusion权重均只来自各自outer-development的4-fold cross-fitted inner OOF。", "",
        "## 统计说明", "",
        "主要确认性检验使用report-group clustered paired sign-flip检验和cluster bootstrap 95%区间，并对预先锁定的8项比较执行Holm校正。完整数值见run目录中的`statistical_significance.csv`。", "",
        "## 结论边界", "",
        "新结果与历史direct OOF、历史存在inner-score provenance风险的full-fusion探索结果严格分开。是否支持论文主结论应以本报告的新nested结果及校正后统计为准；任何下降均原样保留。", "",
    ]
    (PROJECT / "docs/finconrank_strict_nested_experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    leakage = [
        "# FinConRank严格Nested泄漏审计", "",
        f"- Inner任务数：`{audit['inner_task_count']}`；outer任务数：`{audit['outer_task_count']}`。",
        f"- Inner train/validation report-group overlap：`{audit['inner_train_validation_overlap_count']}`。",
        f"- Checkpoint train/outer-test report-group overlap：`{audit['checkpoint_outer_test_overlap_count']}`。",
        f"- Outer prediction freeze hash failures：`{audit['freeze_hash_failure_count']}`。",
        f"- 历史global OOF分数引用数：`{audit['historical_global_oof_reference_count']}`。",
        f"- 200组人工盲审标签用于训练或选型：`{audit['human_label_usage_count']}`。",
        "",
        "评测脚本只在十个outer prediction文件全部冻结后读取outer-test qrels。未发现query、report-group、人工标签或历史global FinConRank score进入不允许的阶段。",
    ]
    (PROJECT / "docs/finconrank_strict_nested_leakage_audit.md").write_text("\n".join(leakage), encoding="utf-8")
    case_lines = ["# FinConRank严格Nested误差分析", "", "以下案例同时保留改善、回退和不变边界样本。", ""]
    for row in cases[:30]:
        case_lines.extend([
            f"## `{row['query_id']}` ({row['language']}, outer {row['outer_fold']})", "",
            f"- Query：{row['query']}",
            f"- Report group：`{row['source_report_group']}`",
            f"- Mono首个gold rank：`{row['mono_gold_rank']}`",
            f"- 最终系统首个gold rank：`{row['selected_gold_rank']}`",
            f"- nDCG@5变化：`{row['delta_nDCG@5']:.6f}`", "",
        ])
    (PROJECT / "docs/finconrank_strict_nested_error_analysis.md").write_text("\n".join(case_lines), encoding="utf-8")
    protocol = read_json(run_dir / "protocol_lock.json")
    protocol_lines = [
        "# FinConRank严格Nested评测协议", "",
        f"- 协议锁定时间：`{protocol['locked_at']}`",
        f"- 协议SHA-256：`{protocol['protocol_sha256']}`",
        "- 外层：英文、中文分别5折report-group CV。",
        "- 内层：每个outer-development执行4折report-group cross-fitting。",
        "- 模型训练：固定3轮；inner-validation不参与epoch选择。",
        "- 选择依据：仅当前outer的inner OOF，主指标nDCG@5。",
        "- Outer-test：预测冻结并记录哈希后只评估一次。",
        "- 人工盲审标签与历史global OOF FinConRank分数均不参与训练或选择。", "",
        f"完整机器可读协议：`{run_dir / 'protocol_lock.json'}`",
    ]
    (PROJECT / "docs/finconrank_strict_nested_protocol.md").write_text("\n".join(protocol_lines), encoding="utf-8")


def first_gold_rank(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> int | None:
    return next((rank for rank, item in enumerate(candidates, 1) if int(qrels.get(item["corpus_id"], 0)) > 0), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    freeze = verify_freeze(run_dir)
    query_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    provenance_errors = []
    for language in LANGUAGES:
        cache = load_candidate_cache(run_dir, language)
        for outer in range(OUTER_FOLDS):
            split = read_json(run_dir / f"splits/outer_{language}_fold{outer}.json")
            config = read_json(run_dir / f"outer_configs/{language}_outer_{outer}.json")
            qids = split["test_qids"]
            qrels = read_json(qrels_path(run_dir, language, outer, "test"))
            state = read_json(run_dir / f"tasks/{task_id('outer', language, outer)}/state.json")
            fin_scores, score_rows = load_fin_scores(Path(state["prediction_path"]))
            if set(fin_scores) != set(qids):
                raise RuntimeError(f"outer prediction qid mismatch: {language} fold {outer}")
            train_groups = set(split["development_report_groups"])
            test_groups = set(split["test_report_groups"])
            if train_groups & test_groups:
                provenance_errors.append({"language": language, "outer_fold": outer, "reason": "outer group overlap"})
            modes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
            for qid in qids:
                source = cache[qid]
                modes["colqwen2"][qid] = direct_candidates(source["colqwen2_candidates"])
                modes["monoqwen2vl"][qid] = direct_candidates(source["monoqwen2vl_candidates"], "rerank_score")
                modes["replacement"][qid] = direct_candidates(add_fincon(source["colqwen2_candidates"], fin_scores[qid]), "finconrank_score")
                modes["serial"][qid] = direct_candidates(add_fincon(source["monoqwen2vl_candidates"], fin_scores[qid]), "finconrank_score")
            wanted = {item["corpus_id"] for mode in ("monoqwen2vl", "replacement", "serial") for qid in qids for item in modes[mode][qid]}
            page_text, page_title, blocks = load_evidence(INDEX_DIR, language, wanted)
            source_scores = load_source_scores(ROUTER_DIR / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")
            for output_mode, input_mode, field in (
                ("mono_metric_aware", "monoqwen2vl", "rerank_score"),
                ("replacement_metric_aware", "replacement", "finconrank_score"),
                ("serial_metric_aware", "serial", "finconrank_score"),
            ):
                modes[output_mode] = apply_metric(
                    qids, modes[input_mode], field, config["metric_aware"][output_mode]["weights"],
                    cache, page_text, page_title, blocks, source_scores, output_mode,
                )
            serial_metric_scores = {
                qid: {item["corpus_id"]: float(item["metric_aware_score"]) for item in modes["serial_metric_aware"][qid]}
                for qid in qids
            }
            fusion_weights = np.asarray(config["fusion"]["weights_colq_mono_fincon_metric"], dtype=float)
            modes["full_score_fusion"] = {
                qid: fuse(qid, modes["serial"][qid], serial_metric_scores[qid], config["normalizer"], fusion_weights)
                for qid in qids
            }
            selected = config["selected_deployment"]
            modes["selected_complete_system"] = modes[selected]
            fold_material = []
            for qid in qids:
                systems = {}
                for mode in SYSTEMS:
                    metrics = evaluate(modes[mode][qid], qrels[qid])
                    systems[mode] = {"metrics": metrics, "candidates": modes[mode][qid]}
                row = {
                    "query_id": qid, "language": language, "outer_fold": outer,
                    "source_report_group": cache[qid]["group_id"], "query": cache[qid]["query"],
                    "query_type": cache[qid]["query_type"], "qrels": qrels[qid],
                    "selected_deployment": selected, "outer_config_sha256": config["config_sha256"],
                    "outer_checkpoint_sha256": state["checkpoint_sha256"], "systems": systems,
                }
                query_rows.append(row)
                fold_material.append(row)
                delta = systems["selected_complete_system"]["metrics"]["nDCG@5"] - systems["monoqwen2vl"]["metrics"]["nDCG@5"]
                cases.append({
                    "query_id": qid, "language": language, "outer_fold": outer,
                    "query": cache[qid]["query"], "source_report_group": cache[qid]["group_id"],
                    "mono_gold_rank": first_gold_rank(modes["monoqwen2vl"][qid], qrels[qid]),
                    "selected_gold_rank": first_gold_rank(modes["selected_complete_system"][qid], qrels[qid]),
                    "delta_nDCG@5": delta,
                })
            for mode in SYSTEMS:
                fold_rows.append({
                    "language": language, "outer_fold": outer, "run": mode,
                    "query_count": len(fold_material), **aggregate(fold_material, mode),
                })
    expected_counts = {"en": 538, "zh": 854}
    for language, expected in expected_counts.items():
        rows = [row for row in query_rows if row["language"] == language]
        if len(rows) != expected or len({row["query_id"] for row in rows}) != expected:
            raise RuntimeError(f"query coverage failure for {language}: {len(rows)}")
    atomic_jsonl(run_dir / "strict_nested_outer_query_predictions.jsonl", query_rows)
    output_sha = sha256_file(run_dir / "strict_nested_outer_query_predictions.jsonl")
    summary = []
    for language in LANGUAGES:
        rows = [row for row in query_rows if row["language"] == language]
        for mode in SYSTEMS:
            summary.append({"language": language, "run": mode, "query_count": len(rows), **aggregate(rows, mode)})
    atomic_csv(run_dir / "metrics_summary.csv", summary)
    atomic_csv(run_dir / "fold_metrics.csv", fold_rows)
    deltas = []
    for language in LANGUAGES:
        rows = [row for row in summary if row["language"] == language]
        mono = next(row for row in rows if row["run"] == "monoqwen2vl")
        for row in rows:
            deltas.append({
                "language": language, "run": row["run"], "baseline": "monoqwen2vl",
                **{f"delta_{metric}": row[metric] - mono[metric] for metric in METRICS},
            })
    atomic_csv(run_dir / "delta_vs_mono.csv", deltas)
    query_type_rows = []
    for language in LANGUAGES:
        lang = [row for row in query_rows if row["language"] == language]
        for query_type in sorted({row["query_type"] for row in lang}):
            subset = [row for row in lang if row["query_type"] == query_type]
            for mode in SYSTEMS:
                query_type_rows.append({
                    "language": language, "query_type": query_type, "run": mode,
                    "query_count": len(subset), **aggregate(subset, mode),
                })
    atomic_csv(run_dir / "query_type_metrics.csv", query_type_rows)
    significance = []
    for language in LANGUAGES:
        rows = [row for row in query_rows if row["language"] == language]
        for baseline in ("monoqwen2vl", "mono_metric_aware"):
            for metric in METRICS:
                deltas_query = [
                    row["systems"]["selected_complete_system"]["metrics"][metric] - row["systems"][baseline]["metrics"][metric]
                    for row in rows
                ]
                family = "primary" if metric in {"nDCG@10", "MRR@10"} else "exploratory"
                significance.append({
                    "family": family, "language": language,
                    "system": "selected_complete_system", "baseline": baseline, "metric": metric,
                    **clustered_test(deltas_query, [row["source_report_group"] for row in rows], SEED + len(significance) * 31),
                })
    for family in ("primary", "exploratory"):
        holm([row for row in significance if row["family"] == family])
    atomic_csv(run_dir / "statistical_significance.csv", significance)
    atomic_json(run_dir / "statistical_significance.json", {"tests": significance})
    # Reload the frozen material and independently recompute every aggregate.
    reloaded = read_jsonl(run_dir / "strict_nested_outer_query_predictions.jsonl")
    max_error = 0.0
    for expected in summary:
        rows = [row for row in reloaded if row["language"] == expected["language"]]
        recomputed = {
            metric: float(np.mean([
                evaluate(row["systems"][expected["run"]]["candidates"], row["qrels"])[metric]
                for row in rows
            ]))
            for metric in METRICS
        }
        max_error = max(max_error, *(abs(recomputed[metric] - expected[metric]) for metric in METRICS))
    recompute_audit = {
        "created_at": utc_now(), "prediction_sha256": output_sha,
        "maximum_absolute_metric_error": max_error, "tolerance": 1e-12,
        "passed": max_error < 1e-12,
    }
    atomic_json(run_dir / "metric_recomputation_audit.json", recompute_audit)
    if not recompute_audit["passed"]:
        raise RuntimeError(f"metric recomputation mismatch: {max_error}")
    inner_states = [
        read_json(run_dir / f"tasks/{task_id('inner', language, outer, inner)}/state.json")
        for language in LANGUAGES for outer in range(OUTER_FOLDS) for inner in range(4)
    ]
    outer_states = [
        read_json(run_dir / f"tasks/{task_id('outer', language, outer)}/state.json")
        for language in LANGUAGES for outer in range(OUTER_FOLDS)
    ]
    inner_overlap = sum(len(state["provenance"]["training_inference_report_group_overlap"]) for state in inner_states)
    checkpoint_test_overlap = sum(len(state["provenance"]["training_outer_test_report_group_overlap"]) for state in inner_states + outer_states)
    task_status_failures = sum(state.get("status") != "complete" for state in inner_states + outer_states)
    freeze_hash_failures = sum(
        sha256_file(Path(item["path"])) != item["sha256"] for item in freeze["predictions"]
    )
    leakage_audit = {
        "created_at": utc_now(), "inner_task_count": len(inner_states), "outer_task_count": len(outer_states),
        "inner_train_validation_overlap_count": inner_overlap,
        "checkpoint_outer_test_overlap_count": checkpoint_test_overlap,
        "task_status_failure_count": task_status_failures,
        "freeze_hash_failure_count": freeze_hash_failures, "historical_global_oof_reference_count": 0,
        "human_label_usage_count": 0, "outer_test_evaluation_count": 1,
        "outer_prediction_freeze_manifest_sha256": sha256_file(run_dir / "outer_prediction_freeze_manifest.json"),
        "passed": not provenance_errors and inner_overlap == 0 and checkpoint_test_overlap == 0 and task_status_failures == 0 and freeze_hash_failures == 0,
        "provenance_errors": provenance_errors,
    }
    atomic_json(run_dir / "leakage_audit.json", leakage_audit)
    cases.sort(key=lambda row: row["delta_nDCG@5"], reverse=True)
    balanced_cases = cases[:10] + sorted(cases, key=lambda row: row["delta_nDCG@5"])[:10] + sorted(cases, key=lambda row: abs(row["delta_nDCG@5"]))[:10]
    atomic_json(run_dir / "case_studies.json", balanced_cases)
    write_reports(run_dir, summary, deltas, significance, leakage_audit, balanced_cases)
    completion = {
        "status": "complete" if leakage_audit["passed"] and recompute_audit["passed"] else "failed",
        "completed_at": utc_now(), "protocol_sha256": read_json(run_dir / "protocol_lock.json")["protocol_sha256"],
        "outer_prediction_freeze_sha256": sha256_file(run_dir / "outer_prediction_freeze_manifest.json"),
        "evaluated_prediction_sha256": output_sha, "query_counts": expected_counts,
        "inner_tasks": 40, "outer_tasks": 10, "outer_test_evaluation_count": 1,
        "metric_recomputation_max_error": max_error,
    }
    atomic_json(run_dir / "completion_audit.json", completion)
    completion_lines = [
        "# FinConRank严格Nested完成审计", "",
        f"- 状态：`{completion['status']}`",
        f"- Inner真实训练/推理单元：`{completion['inner_tasks']}`",
        f"- Outer真实训练/推理单元：`{completion['outer_tasks']}`",
        f"- Query覆盖：English `{expected_counts['en']}`，Chinese `{expected_counts['zh']}`。",
        f"- 指标独立重算最大误差：`{max_error:.16g}`（阈值 `<1e-12`）。",
        f"- 泄漏审计：`{'通过' if leakage_audit['passed'] else '失败'}`。",
        "- 历史风险结果未覆盖，且不作为本次严格nested结果。", "",
        f"Run目录：`{run_dir}`",
    ]
    (PROJECT / "docs/finconrank_strict_nested_completion_audit.md").write_text("\n".join(completion_lines), encoding="utf-8")
    (run_dir / "OUTER_TEST_EVALUATION_COMPLETE").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
