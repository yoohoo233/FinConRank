#!/usr/bin/env python3
"""Official DocReRank direct-reranker baseline on a fixed ColQwen2 top-20 pool."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from evaluate_finconrank_strict_nested import clustered_test, holm, load_fin_scores
from finconrank.common import METRICS, evaluate
from finconrank.strict_nested import (
    IMAGE_ROOTS,
    LANGUAGES,
    OUTER_FOLDS,
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


REVISION = "292fcbdd8bb3ce066cd6c9a0de2d8212ccbd8d3a"
PROMPT_TEMPLATE = "Assert the relevance of the previous image document to the following query, answer True or False. The query is: {query}"
MIN_PIXELS = 200704
MAX_PIXELS = 589824
CANDIDATE_DEPTH = 20


def query_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["language"]), str(row["query_id"])


def comparison_plan() -> list[dict[str, Any]]:
    plan = []
    comparison_index = 0
    for baseline in ("monoqwen2vl", "finconrank"):
        for language in LANGUAGES:
            for metric in ("nDCG@10", "MRR@10"):
                plan.append({
                    "comparison_index": comparison_index,
                    "system": "docrerank",
                    "baseline": baseline,
                    "language": language,
                    "metric": metric,
                    "delta_direction": f"docrerank minus {baseline}",
                    "seed": SEED + comparison_index * 31,
                })
                comparison_index += 1
    return plan


def direct_rank(candidates: list[dict[str, Any]], score_field: str, run_name: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in candidates:
        cid = str(item["corpus_id"])
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            float(item[score_field]),
            -int(item.get("colqwen2_rank", item.get("rank", 999))),
        ),
        reverse=True,
    )
    for rank, item in enumerate(rows, 1):
        item["rank"] = rank
        item["run_name"] = run_name
    return rows


def score_probability(logits: torch.Tensor, true_id: int, false_id: int) -> torch.Tensor:
    return torch.softmax(logits[:, [true_id, false_id]].float(), dim=-1)[:, 0]


def official_messages(query: str, image: Image.Image) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_TEMPLATE.format(query=query)},
        ],
    }]


def command_prepare(strict_run: Path, revision_run: Path, adapter: Path, base_model: Path) -> None:
    result_dir = revision_run / "results" / "docrerank"
    candidate_dir = result_dir / "locked_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "README.md"]
    for name in adapter_files:
        if not (adapter / name).is_file():
            raise RuntimeError(f"missing authorized adapter file: {adapter / name}")
    adapter_config = read_json(adapter / "adapter_config.json")
    readme = (adapter / "README.md").read_text(encoding="utf-8")
    if adapter_config.get("base_model_name_or_path") != "Qwen/Qwen2-VL-2B-Instruct":
        raise RuntimeError("DocReRank adapter declares an unexpected base model")
    if "license: cc-by-4.0" not in readme:
        raise RuntimeError("fixed-revision model card does not state cc-by-4.0")
    manifest_rows = []
    all_rows = []
    for language in LANGUAGES:
        cache = load_candidate_cache(strict_run, language)
        output_rows = []
        for outer in range(OUTER_FOLDS):
            split = read_json(strict_run / "splits" / f"outer_{language}_fold{outer}.json")
            state = read_json(strict_run / "tasks" / task_id("outer", language, outer) / "state.json")
            fin_scores, _ = load_fin_scores(Path(state["prediction_path"]))
            for qid in split["test_qids"]:
                source = cache[qid]
                mono_scores = {
                    item["corpus_id"]: float(item["rerank_score"])
                    for item in source["monoqwen2vl_candidates"]
                    if item.get("rerank_score") is not None
                }
                candidates = []
                for rank, item in enumerate(source["colqwen2_candidates"][:CANDIDATE_DEPTH], 1):
                    cid = item["corpus_id"]
                    image_path = IMAGE_ROOTS[language] / cid
                    if cid not in mono_scores or cid not in fin_scores[qid] or not image_path.is_file():
                        raise RuntimeError(f"same-pool input incomplete: {language}/{qid}/{cid}")
                    candidates.append({
                        "candidate_id": cid,
                        "corpus_id": cid,
                        "colqwen2_rank": rank,
                        "colqwen2_score": float(item["score"]),
                        "monoqwen2vl_score": mono_scores[cid],
                        "finconrank_score": float(fin_scores[qid][cid]),
                        "image_path": str(image_path.resolve()),
                    })
                output_rows.append({
                    "query_id": qid,
                    "language": language,
                    "outer_fold": outer,
                    "source_report_group": source["group_id"],
                    "query": source["query"],
                    "query_type": source["query_type"],
                    "finconrank_checkpoint_sha256": state["checkpoint_sha256"],
                    "candidate_policy": "frozen ColQwen2 top-20 identities",
                    "candidates": candidates,
                    "qrels_included": False,
                })
        output = candidate_dir / f"{language}_colqwen2_top20.jsonl"
        atomic_jsonl(output, output_rows)
        manifest_rows.append({
            "language": language,
            "path": str(output),
            "query_count": len(output_rows),
            "empty_query_count": sum(not row["candidates"] for row in output_rows),
            "candidate_count": sum(len(row["candidates"]) for row in output_rows),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        })
        all_rows.extend(output_rows)
    protocol = {
        "protocol_name": "Official DocReRank same-ColQwen2-pool direct reranker comparison",
        "status": "locked_before_docrerank_gpu_inference",
        "locked_at": utc_now(),
        "revision_driven_not_original_preregistration": True,
        "repository": "DocReRank/DocReRank-Reranker",
        "revision": REVISION,
        "license": "cc-by-4.0 from fixed-revision README",
        "base_model": str(base_model),
        "base_model_commit": base_model.name,
        "adapter": str(adapter),
        "adapter_files": [
            {"name": name, "bytes": (adapter / name).stat().st_size, "sha256": sha256_file(adapter / name)}
            for name in adapter_files
        ],
        "prompt": PROMPT_TEMPLATE,
        "score": "softmax over [True, False] final-token logits; probability of True",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "batch_size": 2,
        "candidate_policy": "exact frozen ColQwen2 top-20 identities for DocReRank, MonoQwen2-VL, and FinConRank",
        "candidate_manifests": manifest_rows,
        "query_count": len(all_rows),
        "candidate_count": sum(len(row["candidates"]) for row in all_rows),
        "empty_query_count": sum(not row["candidates"] for row in all_rows),
        "evaluation": {
            "metrics": METRICS,
            "comparisons": ["DocReRank minus MonoQwen2-VL", "DocReRank minus FinConRank"],
            "languages": list(LANGUAGES),
            "primary_metrics": ["nDCG@10", "MRR@10"],
            "family_size": 8,
            "bootstrap_samples": 10000,
            "sign_flip_samples": 10000,
            "cluster_unit": "source report group",
            "holm_family": "all 8 tests",
            "seed_rule": "42 + comparison_index * 31",
            "comparison_index_map": comparison_plan(),
        },
        "outer_test_qrels_read": False,
        "no_model_selection": True,
        "no_training": True,
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    atomic_json(result_dir / "protocol_lock.json", protocol)
    atomic_csv(result_dir / "candidate_manifest.csv", manifest_rows)
    (result_dir / "PROTOCOL_AND_CANDIDATES_LOCKED").write_text(
        f"{sha256_file(result_dir / 'protocol_lock.json')}  protocol_lock.json\n",
        encoding="ascii",
    )
    print(json.dumps({"status": "prepared", "queries": len(all_rows), "pairs": protocol["candidate_count"]}, indent=2))


def verify_protocol(result_dir: Path) -> dict[str, Any]:
    protocol = read_json(result_dir / "protocol_lock.json")
    if protocol.get("status") != "locked_before_docrerank_gpu_inference":
        raise RuntimeError("DocReRank protocol not locked")
    for entry in protocol["adapter_files"]:
        path = Path(protocol["adapter"]) / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"adapter file changed: {path}")
    for entry in protocol["candidate_manifests"]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"candidate manifest changed: {entry['path']}")
    return protocol


def command_lock_evaluation(revision_run: Path) -> None:
    result_dir = revision_run / "results" / "docrerank"
    protocol = verify_protocol(result_dir)
    output = result_dir / "evaluation_protocol_addendum.json"
    if output.exists():
        raise RuntimeError(f"evaluation addendum already exists: {output}")
    duplicate_rows = []
    total_unique = 0
    for entry in protocol["candidate_manifests"]:
        for row in read_jsonl(Path(entry["path"])):
            ids = [str(item["corpus_id"]) for item in row["candidates"]]
            unique = list(dict.fromkeys(ids))
            total_unique += len(unique)
            if len(unique) != len(ids):
                duplicate_rows.append({
                    "language": row["language"], "query_id": row["query_id"],
                    "candidate_rows": len(ids), "unique_corpus_ids": len(unique),
                    "duplicates_removed_for_evaluation": len(ids) - len(unique),
                })
    addendum = {
        "status": "locked_before_outer_test_qrels_evaluation",
        "locked_at": utc_now(),
        "reason": "The corrected strict evaluator follows official TREC dictionary semantics: duplicate corpus IDs are not independent retrieval results.",
        "inference_protocol_path": str(result_dir / "protocol_lock.json"),
        "inference_protocol_sha256": sha256_file(result_dir / "protocol_lock.json"),
        "evaluation_script_path": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "deduplication": "before score sorting, preserve the first occurrence in the locked ColQwen2 candidate order for each corpus_id",
        "raw_candidate_rows": protocol["candidate_count"],
        "unique_candidate_rows": total_unique,
        "affected_queries": duplicate_rows,
        "comparisons": protocol["evaluation"]["comparison_index_map"],
        "outer_test_qrels_read": False,
        "model_scores_changed": False,
        "model_inference_repeated": False,
    }
    addendum["addendum_sha256"] = stable_hash(addendum)
    atomic_json(output, addendum)
    (result_dir / "EVALUATION_PROTOCOL_LOCKED").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="ascii"
    )
    print(json.dumps({"status": "evaluation_locked", "affected_queries": len(duplicate_rows), "unique_pairs": total_unique}, indent=2))


def verify_evaluation_addendum(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "evaluation_protocol_addendum.json"
    addendum = read_json(path)
    if addendum.get("status") != "locked_before_outer_test_qrels_evaluation":
        raise RuntimeError("DocReRank evaluation addendum is not locked")
    if sha256_file(result_dir / "protocol_lock.json") != addendum["inference_protocol_sha256"]:
        raise RuntimeError("DocReRank inference protocol changed after evaluation lock")
    if sha256_file(Path(__file__).resolve()) != addendum["evaluation_script_sha256"]:
        raise RuntimeError("DocReRank evaluation script changed after evaluation lock")
    return addendum


def load_model(protocol: dict[str, Any]) -> tuple[Any, Any, int, int, float]:
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(protocol["base_model"], local_files_only=True)
    processor.image_processor.min_pixels = int(protocol["min_pixels"])
    processor.image_processor.max_pixels = int(protocol["max_pixels"])
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        protocol["base_model"], torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation=protocol["attention"],
    )
    model = PeftModel.from_pretrained(base, protocol["adapter"], local_files_only=True).to("cuda:0").eval()
    true_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_id = processor.tokenizer.convert_tokens_to_ids("False")
    if true_id < 0 or false_id < 0 or true_id == false_id:
        raise RuntimeError(f"invalid True/False token IDs: {true_id}/{false_id}")
    return processor, model, true_id, false_id, time.perf_counter() - started


@torch.inference_mode()
def score_batch(
    processor: Any,
    model: Any,
    true_id: int,
    false_id: int,
    query: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[float], dict[str, float]]:
    decode_started = time.perf_counter()
    images = []
    for candidate in candidates:
        with Image.open(candidate["image_path"]) as image:
            images.append(image.convert("RGB").copy())
    decode_seconds = time.perf_counter() - decode_started
    preprocess_started = time.perf_counter()
    texts = [
        processor.apply_chat_template(official_messages(query, image), tokenize=False, add_generation_prompt=True)
        for image in images
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    preprocess_seconds = time.perf_counter() - preprocess_started
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    logits = model(**inputs).logits[:, -1, :]
    scores = score_probability(logits, true_id, false_id)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    return [float(value) for value in scores.cpu().tolist()], {
        "image_decode_seconds": decode_seconds,
        "preprocess_seconds": preprocess_seconds,
        "gpu_inference_seconds": inference_seconds,
    }


def command_worker(
    revision_run: Path,
    worker_index: int,
    num_workers: int,
    output: Path,
    stats_path: Path,
    max_queries: int | None = None,
) -> None:
    result_dir = revision_run / "results" / "docrerank"
    protocol = verify_protocol(result_dir)
    rows = []
    for entry in protocol["candidate_manifests"]:
        rows.extend(read_jsonl(Path(entry["path"])))
    assigned = [row for index, row in enumerate(rows) if index % num_workers == worker_index]
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        assigned = assigned[:max_queries]
    torch.cuda.reset_peak_memory_stats()
    processor, model, true_id, false_id, load_seconds = load_model(protocol)
    output_rows = []
    totals = defaultdict(float)
    scored_pairs = 0
    batch_size = int(protocol["batch_size"])
    wall_started = time.perf_counter()
    for query_index, row in enumerate(assigned, 1):
        candidates = [dict(item) for item in row["candidates"]]
        query_timing = defaultdict(float)
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset: offset + batch_size]
            scores, timing = score_batch(processor, model, true_id, false_id, row["query"], batch)
            for item, score in zip(batch, scores):
                item["docrerank_score"] = score
            for key, value in timing.items():
                query_timing[key] += value
                totals[key] += value
            scored_pairs += len(batch)
        output_rows.append({
            **{key: value for key, value in row.items() if key != "candidates"},
            "worker_index": worker_index,
            "num_workers": num_workers,
            "physical_gpu_id": int(os.environ.get("FINCONRANK_PHYSICAL_GPU_ID", "-1")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inference_timestamp": utc_now(),
            "query_timing": dict(query_timing),
            "candidates": candidates,
        })
        if query_index % 20 == 0 or query_index == len(assigned):
            print(json.dumps({"worker": worker_index, "queries": query_index, "total": len(assigned), "pairs": scored_pairs}), flush=True)
    atomic_jsonl(output, output_rows)
    stats = {
        "worker_index": worker_index, "num_workers": num_workers,
        "physical_gpu_id": int(os.environ.get("FINCONRANK_PHYSICAL_GPU_ID", "-1")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "queries": len(output_rows), "scored_pairs": scored_pairs,
        "model_load_seconds": load_seconds,
        **totals,
        "worker_wall_seconds": time.perf_counter() - wall_started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "output": str(output), "output_sha256": sha256_file(output),
        "finished_at": utc_now(), "status": "complete",
    }
    atomic_json(stats_path, stats)


def command_merge(revision_run: Path, worker_outputs: list[Path]) -> None:
    result_dir = revision_run / "results" / "docrerank"
    protocol = verify_protocol(result_dir)
    expected = []
    for entry in protocol["candidate_manifests"]:
        expected.extend(read_jsonl(Path(entry["path"])))
    expected_ids = [query_key(row) for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("duplicate language/query key in locked candidates")
    observed = {}
    for path in worker_outputs:
        for row in read_jsonl(path):
            key = query_key(row)
            if key in observed:
                raise RuntimeError(f"duplicate worker query: {key}")
            observed[key] = row
    if set(observed) != set(expected_ids):
        raise RuntimeError(f"worker coverage mismatch: missing={len(set(expected_ids)-set(observed))} extra={len(set(observed)-set(expected_ids))}")
    rows = [observed[key] for key in expected_ids]
    for row in rows:
        if any(item.get("docrerank_score") is None or not np.isfinite(item["docrerank_score"]) for item in row["candidates"]):
            raise RuntimeError(f"invalid DocReRank score: {row['query_id']}")
    output = result_dir / "docrerank_frozen_predictions.jsonl"
    atomic_jsonl(output, rows)
    freeze = {
        "status": "frozen_before_outer_test_qrels_evaluation",
        "created_at": utc_now(), "path": str(output), "bytes": output.stat().st_size,
        "sha256": sha256_file(output), "queries": len(rows),
        "candidate_pairs": sum(len(row["candidates"]) for row in rows),
        "worker_outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in worker_outputs
        ],
        "outer_test_metrics_seen": False,
    }
    atomic_json(result_dir / "prediction_freeze_manifest.json", freeze)
    (result_dir / "PREDICTIONS_FROZEN").write_text(f"{freeze['sha256']}  {output.name}\n", encoding="ascii")
    print(json.dumps({"status": "frozen", "queries": len(rows), "pairs": freeze["candidate_pairs"]}, indent=2))


def command_evaluate(strict_run: Path, revision_run: Path) -> None:
    result_dir = revision_run / "results" / "docrerank"
    protocol = verify_protocol(result_dir)
    addendum = verify_evaluation_addendum(result_dir)
    freeze = read_json(result_dir / "prediction_freeze_manifest.json")
    prediction_path = Path(freeze["path"])
    if freeze.get("status") != "frozen_before_outer_test_qrels_evaluation" or sha256_file(prediction_path) != freeze["sha256"]:
        raise RuntimeError("DocReRank predictions are not validly frozen")
    if (result_dir / "OUTER_TEST_EVALUATION_COMPLETE").exists():
        raise RuntimeError("DocReRank outer test already evaluated")
    qrels_by_fold = {
        (language, outer): read_json(qrels_path(strict_run, language, outer, "test"))
        for language in LANGUAGES for outer in range(OUTER_FOLDS)
    }
    query_rows = []
    for row in read_jsonl(prediction_path):
        qrels = qrels_by_fold[(row["language"], int(row["outer_fold"]))][row["query_id"]]
        systems = {
            "colqwen2": direct_rank(row["candidates"], "colqwen2_score", "colqwen2_same_pool"),
            "monoqwen2vl": direct_rank(row["candidates"], "monoqwen2vl_score", "monoqwen2vl_same_pool"),
            "finconrank": direct_rank(row["candidates"], "finconrank_score", "finconrank_same_pool"),
            "docrerank": direct_rank(row["candidates"], "docrerank_score", "docrerank_same_pool"),
        }
        query_rows.append({
            "query_id": row["query_id"], "language": row["language"], "outer_fold": row["outer_fold"],
            "source_report_group": row["source_report_group"], "query_type": row["query_type"], "qrels": qrels,
            "systems": {name: {"metrics": evaluate(items, qrels), "candidates": items} for name, items in systems.items()},
        })
    for language, expected in (("en", 538), ("zh", 854)):
        rows = [row for row in query_rows if row["language"] == language]
        if len(rows) != expected or len({row["query_id"] for row in rows}) != expected:
            raise RuntimeError(f"DocReRank query coverage mismatch: {language}/{len(rows)}")
    atomic_jsonl(result_dir / "query_level_evaluated_predictions.jsonl", query_rows)
    metrics_rows = []
    fold_rows = []
    query_type_rows = []
    for language in LANGUAGES:
        rows = [row for row in query_rows if row["language"] == language]
        for system in ("colqwen2", "monoqwen2vl", "finconrank", "docrerank"):
            metrics_rows.append({
                "language": language, "run": system, "query_count": len(rows),
                **{metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in rows])) for metric in METRICS},
            })
            for outer_fold in range(OUTER_FOLDS):
                fold_subset = [row for row in rows if int(row["outer_fold"]) == outer_fold]
                fold_rows.append({
                    "language": language, "outer_fold": outer_fold, "run": system,
                    "query_count": len(fold_subset),
                    **{metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in fold_subset])) for metric in METRICS},
                })
            for query_type in sorted({row["query_type"] for row in rows}):
                type_subset = [row for row in rows if row["query_type"] == query_type]
                query_type_rows.append({
                    "language": language, "query_type": query_type, "run": system,
                    "query_count": len(type_subset),
                    **{metric: float(np.mean([row["systems"][system]["metrics"][metric] for row in type_subset])) for metric in METRICS},
                })
    atomic_csv(result_dir / "metrics_summary.csv", metrics_rows)
    atomic_csv(result_dir / "fold_metrics.csv", fold_rows)
    atomic_csv(result_dir / "query_type_metrics.csv", query_type_rows)
    delta_rows = []
    for language in LANGUAGES:
        language_metrics = {row["run"]: row for row in metrics_rows if row["language"] == language}
        for baseline in ("monoqwen2vl", "finconrank"):
            delta_rows.append({
                "language": language, "run": "docrerank", "baseline": baseline,
                **{f"delta_{metric}": language_metrics["docrerank"][metric] - language_metrics[baseline][metric] for metric in METRICS},
            })
    atomic_csv(result_dir / "delta_vs_direct_rerankers.csv", delta_rows)
    significance = []
    for comparison in protocol["evaluation"]["comparison_index_map"]:
        baseline = comparison["baseline"]
        language = comparison["language"]
        metric = comparison["metric"]
        rows = [row for row in query_rows if row["language"] == language]
        deltas = [row["systems"]["docrerank"]["metrics"][metric] - row["systems"][baseline]["metrics"][metric] for row in rows]
        significance.append({
            "family": "revision_docrerank_eight_tests",
            **comparison,
            **clustered_test(deltas, [row["source_report_group"] for row in rows], comparison["seed"], samples=10000),
        })
    holm(significance)
    atomic_csv(result_dir / "statistical_significance.csv", significance)
    atomic_json(result_dir / "statistical_significance.json", {"tests": significance})
    audit = {
        "created_at": utc_now(), "protocol_sha256": sha256_file(result_dir / "protocol_lock.json"),
        "frozen_prediction_sha256": freeze["sha256"],
        "evaluated_prediction_sha256": sha256_file(result_dir / "query_level_evaluated_predictions.jsonl"),
        "evaluation_addendum_sha256": sha256_file(result_dir / "evaluation_protocol_addendum.json"),
        "queries": len(query_rows), "raw_candidate_pairs": freeze["candidate_pairs"],
        "unique_candidate_pairs": addendum["unique_candidate_rows"], "tests": len(significance),
        "metric_range_pass": all(
            -1e-12 <= row[metric] <= 1.0 + 1e-12
            for row in metrics_rows for metric in METRICS
        ),
    }
    audit["status"] = "PASS" if audit["metric_range_pass"] else "FAIL"
    atomic_json(result_dir / "evaluation_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"DocReRank evaluation audit failed: {audit}")
    (result_dir / "OUTER_TEST_EVALUATION_COMPLETE").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "evaluated", "queries": len(query_rows), "tests": len(significance)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--strict-run", type=Path, required=True)
    prepare.add_argument("--revision-run", type=Path, required=True)
    prepare.add_argument("--adapter", type=Path, required=True)
    prepare.add_argument("--base-model", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--revision-run", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--num-workers", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--stats", type=Path, required=True)
    worker.add_argument("--max-queries", type=int)
    lock_evaluation = sub.add_parser("lock-evaluation")
    lock_evaluation.add_argument("--revision-run", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--revision-run", type=Path, required=True)
    merge.add_argument("--worker-outputs", type=Path, nargs="+", required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--strict-run", type=Path, required=True)
    evaluate_parser.add_argument("--revision-run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        command_prepare(args.strict_run.resolve(), args.revision_run.resolve(), args.adapter.resolve(), args.base_model.resolve())
    elif args.command == "worker":
        command_worker(
            args.revision_run.resolve(), args.worker_index, args.num_workers,
            args.output.resolve(), args.stats.resolve(), args.max_queries,
        )
    elif args.command == "lock-evaluation":
        command_lock_evaluation(args.revision_run.resolve())
    elif args.command == "merge":
        command_merge(args.revision_run.resolve(), [path.resolve() for path in args.worker_outputs])
    else:
        command_evaluate(args.strict_run.resolve(), args.revision_run.resolve())


if __name__ == "__main__":
    main()
