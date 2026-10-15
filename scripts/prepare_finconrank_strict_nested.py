#!/usr/bin/env python3
"""Lock and materialize the leakage-safe FinConRank nested-CV protocol."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finconrank.strict_nested import (
    BASE_MODEL,
    CANDIDATE_DEPTH,
    CONSTRAINTS,
    INNER_FOLDS,
    INITIAL_ADAPTER,
    LANGUAGES,
    OUTER_FOLDS,
    PROJECT,
    SEED,
    SOURCE_OOF,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    candidate_cache_path,
    qrels_path,
    read_json,
    read_jsonl,
    sha256_file,
    source_paths,
    stable_hash,
    task_id,
    utc_now,
)


CODE_PATHS = [
    PROJECT / "configs/finconrank/strict_nested_group_oof.yaml",
    PROJECT / "src/finconrank/strict_nested.py",
    PROJECT / "scripts/prepare_finconrank_strict_nested.py",
    PROJECT / "scripts/run_finconrank_strict_nested_task.py",
    PROJECT / "scripts/select_finconrank_strict_nested.py",
    PROJECT / "scripts/evaluate_finconrank_strict_nested.py",
    PROJECT / "scripts/finconrank_strict_nested_scheduler.py",
    PROJECT / "scripts/train_finconrank.py",
    PROJECT / "scripts/build_finconrank_training_data.py",
    PROJECT / "scripts/evaluate_finconrank.py",
    PROJECT / "scripts/run_financial_metric_aware_reranking.py",
]


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def model_records(root: Path) -> list[dict[str, Any]]:
    patterns = ("*.json", "*.safetensors", "*.model", "*.txt")
    paths = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    return [file_record(path) for path in paths]


def lock_protocol(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "frozen_inputs", "sealed_qrels", "splits", "tasks", "checkpoints",
        "inner_predictions", "outer_predictions", "logs", "gpu_snapshots", "statistics",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in [*source_paths().values(), *CODE_PATHS, BASE_MODEL, INITIAL_ADAPTER] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"protocol inputs missing: {missing}")
    inputs = {name: file_record(path) for name, path in source_paths().items()}
    code = [file_record(path) for path in CODE_PATHS]
    protocol = {
        "protocol_name": "FinConRank strict report-group nested cross-validation",
        "status": "locked_before_new_outer_test_evaluation",
        "locked_at": utc_now(),
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds_per_outer": INNER_FOLDS,
        "languages": list(LANGUAGES),
        "outer_group_source": str((SOURCE_OOF / "oof_manifest.json").resolve()),
        "candidate_depth": CANDIDATE_DEPTH,
        "candidate_policy": {
            "colqwen2": "frozen tab-safe ColQwen2 top-20",
            "monoqwen2vl": "frozen MonoQwen2-VL top-20",
            "finconrank_inference": "union of both candidate sets, scored once per checkpoint",
            "serial_order": "ColQwen2 -> MonoQwen2-VL -> FinConRank",
        },
        "model_recipe": {
            "base_model": str(BASE_MODEL),
            "initial_adapter": str(INITIAL_ADAPTER),
            "variant": "full",
            "objective": "pairwise margin ranking",
            "epochs": 3,
            "checkpoint_for_scoring": "epoch_3_adapter (fixed; no validation-based epoch selection)",
            "learning_rate": 2e-5,
            "margin": 0.2,
            "micro_batch_size": 2,
            "gradient_accumulation": 2,
            "effective_pair_batch_size": 4,
            "oom_retry": {"micro_batch_size": 1, "gradient_accumulation": 4},
            "training_max_pixels": 200704,
            "inference_batch_size": 2,
            "inference_max_pixels": 602112,
            "curriculum": True,
            "constraints_in_prompt": True,
            "false_negative_verification": True,
            "max_pairs_per_query": 6,
        },
        "selection": {
            "scope": "current outer fold cross-fitted inner OOF only",
            "primary_metric": "nDCG@5",
            "tie_break_metrics": ["nDCG@10", "MRR@10"],
            "metric_aware_weight_candidates": "fixed candidate_weights(dim=12, seed=42 + language_offset + outer_fold*100)",
            "fusion_grid": "weights in {0,.2,.4,.6,.8,1}, sum=1, FinConRank weight > 0",
            "normalization": "outer-development inner-OOF global mean/std followed by sigmoid",
            "selectable_system_order_for_final_tie_break": [
                "full_score_fusion", "serial_metric_aware", "replacement_metric_aware",
                "serial", "replacement", "mono_metric_aware",
            ],
            "outer_test_evaluations": 1,
        },
        "reported_systems": [
            "colqwen2", "monoqwen2vl", "replacement", "serial",
            "mono_metric_aware", "replacement_metric_aware", "serial_metric_aware",
            "full_score_fusion", "selected_complete_system",
        ],
        "metrics": ["nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10"],
        "statistics": {
            "unit": "query, resampled/sign-flipped by source report-group cluster",
            "bootstrap_samples": 10000,
            "permutation_samples": 10000,
            "confidence_interval": 0.95,
            "primary_family": "selected complete system vs Mono and vs Mono+Metric-Aware; English/Chinese; nDCG@10/MRR@10 (8 tests)",
            "multiple_comparison": "Holm-Bonferroni",
        },
        "leakage_guards": {
            "human_audit_used": False,
            "historical_global_oof_assembled_scores_allowed": False,
            "inner_model_may_score_own_training_query": False,
            "outer_test_qrels_for_selection": False,
            "outer_test_metrics_before_prediction_freeze": False,
        },
        "outer_test_metrics_seen_for_this_run": False,
        "input_files": inputs,
        "code_files": code,
        "base_model_files": model_records(BASE_MODEL),
        "initial_adapter_files": model_records(INITIAL_ADAPTER),
        "downloads": [],
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    atomic_json(run_dir / "protocol_lock.json", protocol)
    atomic_json(run_dir / "progress.json", {
        "run_dir": str(run_dir.resolve()), "created_at": protocol["locked_at"],
        "phase": "protocol_locked", "status": "preparing", "outer_test_metrics_computed": False,
        "bootstrap_inner_unit_complete": False, "scheduler_pid": None,
    })
    print(json.dumps({"run_dir": str(run_dir), "protocol_sha256": protocol["protocol_sha256"]}, indent=2))


def balanced_group_folds(groups: list[str], qids_by_group: dict[str, list[str]], folds: int, seed: int) -> list[list[str]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    order = {group: index for index, group in enumerate(shuffled)}
    shuffled.sort(key=lambda group: (-len(qids_by_group[group]), order[group]))
    result: list[list[str]] = [[] for _ in range(folds)]
    totals = [0] * folds
    for group in shuffled:
        target = min(range(folds), key=lambda idx: (totals[idx], len(result[idx]), idx))
        result[target].append(group)
        totals[target] += len(qids_by_group[group])
    return [sorted(values) for values in result]


def prepare(run_dir: Path) -> None:
    protocol = read_json(run_dir / "protocol_lock.json")
    if protocol.get("status") != "locked_before_new_outer_test_evaluation":
        raise RuntimeError("protocol is not locked")
    constraints = {row["query_id"]: row for row in read_jsonl(CONSTRAINTS)}
    old_outer = read_json(SOURCE_OOF / "oof_manifest.json")
    old_outer_groups = {int(fold["fold"]): set(fold["test_group_ids"]) for fold in old_outer["folds"]}
    split_manifest: dict[str, Any] = {
        "created_at": utc_now(), "seed": SEED,
        "strategy": "fixed historical outer connected-report-group folds; deterministic balanced 4-fold inner report-group CV within each outer development set",
        "outer_source_sha256": sha256_file(SOURCE_OOF / "oof_manifest.json"),
        "languages": {}, "audits": [],
    }
    all_tasks: list[dict[str, Any]] = []
    for language in LANGUAGES:
        colq_rows = read_jsonl(source_paths()[f"colqwen2_{language}"])
        mono_rows = {row["query_id"]: row for row in read_jsonl(source_paths()[f"monoqwen2vl_{language}"])}
        if len(colq_rows) != len({row["query_id"] for row in colq_rows}):
            raise RuntimeError(f"duplicate query IDs in {language} ColQwen2 source")
        if set(mono_rows) != {row["query_id"] for row in colq_rows}:
            raise RuntimeError(f"ColQwen2/Mono query mismatch for {language}")
        cache_rows = []
        qrels_all: dict[str, dict[str, int]] = {}
        qids_by_group: dict[str, list[str]] = defaultdict(list)
        for source in colq_rows:
            qid = source["query_id"]
            info = constraints[qid]
            if info["language"] != language:
                raise RuntimeError(f"constraint language mismatch: {qid}")
            qids_by_group[info["group_id"]].append(qid)
            qrels_all[qid] = {str(cid): int(rel) for cid, rel in source.get("qrels", {}).items()}
            cache_rows.append({
                "query_id": qid, "language": language, "query": source.get("query", ""),
                "group_id": info["group_id"], "query_type": info["query_type"],
                "constraints": info["constraints"],
                "colqwen2_candidates": source.get("candidates", []),
                "monoqwen2vl_candidates": mono_rows[qid].get("candidates", []),
            })
        atomic_jsonl(candidate_cache_path(run_dir, language), cache_rows)
        language_manifest = {"query_count": len(cache_rows), "report_group_count": len(qids_by_group), "outer_folds": []}
        all_groups = set(qids_by_group)
        for outer in range(OUTER_FOLDS):
            test_groups = old_outer_groups[outer] & all_groups
            development_groups = all_groups - test_groups
            test_qids = sorted(qid for group in test_groups for qid in qids_by_group[group])
            development_qids = sorted(qid for group in development_groups for qid in qids_by_group[group])
            inner_group_folds = balanced_group_folds(
                sorted(development_groups), qids_by_group, INNER_FOLDS,
                SEED + (0 if language == "en" else 10000) + outer * 100,
            )
            inner_rows = []
            for inner, validation_groups_list in enumerate(inner_group_folds):
                validation_groups = set(validation_groups_list)
                train_groups = development_groups - validation_groups
                train_qids = sorted(qid for group in train_groups for qid in qids_by_group[group])
                validation_qids = sorted(qid for group in validation_groups for qid in qids_by_group[group])
                if train_groups & validation_groups or (train_groups | validation_groups) != development_groups:
                    raise RuntimeError(f"inner group partition failure {language} outer={outer} inner={inner}")
                identifier = task_id("inner", language, outer, inner)
                atomic_json(run_dir / f"splits/{identifier}.json", {
                    "task_id": identifier, "kind": "inner", "language": language,
                    "outer_fold": outer, "inner_fold": inner,
                    "train_report_groups": sorted(train_groups),
                    "validation_report_groups": sorted(validation_groups),
                    "outer_test_report_groups": sorted(test_groups),
                    "train_qids": train_qids, "validation_qids": validation_qids,
                    "train_groups_sha256": stable_hash(sorted(train_groups)),
                    "validation_groups_sha256": stable_hash(sorted(validation_groups)),
                    "outer_test_groups_sha256": stable_hash(sorted(test_groups)),
                })
                inner_rows.append({
                    "inner_fold": inner, "train_report_group_count": len(train_groups),
                    "validation_report_group_count": len(validation_groups),
                    "train_query_count": len(train_qids), "validation_query_count": len(validation_qids),
                    "train_validation_group_overlap": 0,
                    "train_outer_test_group_overlap": len(train_groups & test_groups),
                    "validation_outer_test_group_overlap": len(validation_groups & test_groups),
                    "split_file": str((run_dir / f"splits/{identifier}.json").resolve()),
                })
                all_tasks.append({
                    "task_id": identifier, "kind": "inner", "language": language,
                    "outer_fold": outer, "inner_fold": inner, "status": "pending",
                    "attempts": 0, "micro_batch_size": 2, "gradient_accumulation": 2,
                })
            atomic_json(qrels_path(run_dir, language, outer, "development"), {qid: qrels_all[qid] for qid in development_qids})
            atomic_json(qrels_path(run_dir, language, outer, "test"), {qid: qrels_all[qid] for qid in test_qids})
            outer_split = {
                "task_id": task_id("outer", language, outer), "kind": "outer",
                "language": language, "outer_fold": outer,
                "development_report_groups": sorted(development_groups),
                "test_report_groups": sorted(test_groups),
                "development_qids": development_qids, "test_qids": test_qids,
                "development_groups_sha256": stable_hash(sorted(development_groups)),
                "test_groups_sha256": stable_hash(sorted(test_groups)),
                "inner_folds": inner_rows,
            }
            atomic_json(run_dir / f"splits/outer_{language}_fold{outer}.json", outer_split)
            language_manifest["outer_folds"].append(outer_split)
            split_manifest["audits"].append({
                "language": language, "outer_fold": outer,
                "development_query_count": len(development_qids), "test_query_count": len(test_qids),
                "development_report_group_count": len(development_groups), "test_report_group_count": len(test_groups),
                "outer_report_group_overlap": len(development_groups & test_groups),
                "inner_validation_qids_cover_development_once": Counter(
                    qid for fold in inner_rows for qid in read_json(Path(fold["split_file"]))["validation_qids"]
                ) == Counter({qid: 1 for qid in development_qids}),
            })
        split_manifest["languages"][language] = language_manifest
    if sum(split_manifest["languages"][lang]["query_count"] for lang in LANGUAGES) != 1392:
        raise RuntimeError("expected 1,392 benchmark qrel IDs")
    if any(row["outer_report_group_overlap"] for row in split_manifest["audits"]):
        raise RuntimeError("outer report-group overlap")
    if not all(row["inner_validation_qids_cover_development_once"] for row in split_manifest["audits"]):
        raise RuntimeError("inner validation coverage failure")
    atomic_json(run_dir / "split_manifest.json", split_manifest)
    atomic_csv(run_dir / "split_overlap_audit.csv", split_manifest["audits"])
    atomic_json(run_dir / "task_manifest.json", {
        "created_at": utc_now(), "status": "inner_pending", "tasks": all_tasks,
        "required_inner_tasks": 40, "required_outer_tasks": 10,
    })
    progress = read_json(run_dir / "progress.json")
    progress.update({
        "phase": "inner", "status": "ready", "prepared_at": utc_now(),
        "inner_tasks": {"total": 40, "pending": 40, "running": 0, "complete": 0, "failed": 0},
        "outer_tasks": {"total": 10, "pending": 10, "running": 0, "complete": 0, "failed": 0},
        "query_counts": {lang: split_manifest["languages"][lang]["query_count"] for lang in LANGUAGES},
    })
    atomic_json(run_dir / "progress.json", progress)
    reproduction = f"""# Reproduction Commands

All commands run from `<repository-root>` in the `finragbench` environment.

```bash
export PYTHONPATH=src:.
export HF_ENDPOINT=https://hf-mirror.com
RUN_DIR={run_dir}
python scripts/prepare_finconrank_strict_nested.py lock --run-dir \"$RUN_DIR\"
python scripts/prepare_finconrank_strict_nested.py prepare --run-dir \"$RUN_DIR\"
python scripts/finconrank_strict_nested_scheduler.py --run-dir \"$RUN_DIR\"
```

Progress and resume:

```bash
cat \"$RUN_DIR/progress.json\"
python scripts/finconrank_strict_nested_scheduler.py --run-dir \"$RUN_DIR\"
```
"""
    (run_dir / "reproduction_commands.md").write_text(reproduction, encoding="utf-8")
    (run_dir / "model_downloads.md").write_text(
        "# Model Downloads\n\nNo new model or dataset downloads were required. The run uses the existing local Qwen2-VL-2B-Instruct and MonoQwen2-VL adapter snapshots recorded in `protocol_lock.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "prepared", "inner_tasks": 40, "query_counts": progress["query_counts"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("--run-dir", type=Path, required=True)
    materialize = sub.add_parser("prepare")
    materialize.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        lock_protocol(args.run_dir.resolve())
    else:
        prepare(args.run_dir.resolve())


if __name__ == "__main__":
    main()
