#!/usr/bin/env python3
"""Build, train, and infer one strict FinConRank inner or outer unit."""
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from build_finconrank_training_data import choose_pairs, load_selected_text
from train_finconrank import constraint_text, message
from finconrank.strict_nested import (
    BASE_MODEL,
    CANDIDATE_DEPTH,
    IMAGE_ROOTS,
    INDEX_DIR,
    INITIAL_ADAPTER,
    PROJECT,
    SEED,
    atomic_json,
    atomic_jsonl,
    load_candidate_cache,
    qrels_path,
    read_json,
    read_jsonl,
    sha256_file,
    stable_hash,
    task_dir,
    union_candidates,
    utc_now,
)


def deterministic_rng(qid: str) -> random.Random:
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return random.Random(SEED + int(digest[:12], 16))


def write_gpu_snapshot(path: Path) -> None:
    result = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout, encoding="utf-8")


def build_training_pairs(run_dir: Path, identifier: str, split: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    language = split["language"]
    if split["kind"] == "inner":
        train_qids = split["train_qids"]
    else:
        train_qids = split["development_qids"]
    allowed_qrels = read_json(qrels_path(run_dir, language, split["outer_fold"], "development"))
    if not set(train_qids) <= set(allowed_qrels):
        raise RuntimeError("training qids are not contained in outer-development qrels")
    cache = load_candidate_cache(run_dir, language)
    wanted = {
        candidate["corpus_id"]
        for qid in train_qids
        for candidate in cache[qid]["colqwen2_candidates"]
    }
    wanted.update(cid for qid in train_qids for cid in allowed_qrels[qid])
    texts = load_selected_text(INDEX_DIR / f"{language}_pages.jsonl", wanted)
    pairs: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    missing_gold = []
    for qid in sorted(train_qids):
        source = cache[qid]
        row = {
            "query_id": qid,
            "query": source["query"],
            "qrels": allowed_qrels[qid],
            "candidates": source["colqwen2_candidates"],
        }
        constraints_row = {
            "group_id": source["group_id"], "query_type": source["query_type"],
            "constraints": source["constraints"],
        }
        selected, rejected = choose_pairs(
            row, constraints_row, texts, language, "train", deterministic_rng(qid),
            IMAGE_ROOTS["en"], IMAGE_ROOTS["zh"], 6, True,
        )
        pairs.extend(selected)
        uncertain.extend(rejected)
        for cid in allowed_qrels[qid]:
            image = IMAGE_ROOTS[language] / cid
            if not image.exists():
                missing_gold.append({"query_id": qid, "candidate_id": cid, "path": str(image)})
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_dir / "train_pairs.jsonl", pairs)
    atomic_jsonl(output_dir / "empty_dev_pairs.jsonl", [])
    atomic_jsonl(output_dir / "uncertain_pairs.jsonl", uncertain)
    train_groups = sorted({cache[qid]["group_id"] for qid in train_qids})
    type_counts = Counter(kind for row in pairs for kind in row.get("negative_types", []))
    stage_counts = Counter(int(row.get("curriculum_stage", 2)) for row in pairs)
    summary = {
        "created_at": utc_now(), "task_id": identifier, "language": language,
        "outer_fold": split["outer_fold"], "inner_fold": split.get("inner_fold"),
        "training_query_count": len(train_qids), "training_pair_count": len(pairs),
        "training_report_group_count": len(train_groups),
        "training_report_groups": train_groups,
        "training_report_groups_sha256": stable_hash(train_groups),
        "uncertain_excluded": len(uncertain), "missing_gold_images": missing_gold,
        "negative_type_counts": dict(type_counts),
        "curriculum_stage_counts": {str(key): value for key, value in stage_counts.items()},
        "annotation_source": "automatic rule and allowed outer-development qrels only",
        "human_audit_labels_used": False,
    }
    atomic_json(output_dir / "training_data_manifest.json", summary)
    return summary


def train_adapter(
    training_dir: Path,
    checkpoint_dir: Path,
    micro_batch_size: int,
    gradient_accumulation: int,
    seed: int,
) -> Path:
    expected = checkpoint_dir / "epoch_3_adapter/adapter_model.safetensors"
    if expected.exists() and (checkpoint_dir / "training_summary.json").exists():
        return expected.parent
    command = [
        sys.executable, str(PROJECT / "scripts/train_finconrank.py"),
        "--train-pairs", str(training_dir / "train_pairs.jsonl"),
        "--dev-pairs", str(training_dir / "empty_dev_pairs.jsonl"),
        "--output-dir", str(checkpoint_dir),
        "--base-model", str(BASE_MODEL), "--init-adapter", str(INITIAL_ADAPTER),
        "--variant", "full", "--epochs", "3", "--learning-rate", "0.00002",
        "--margin", "0.2", "--batch-size", str(micro_batch_size),
        "--eval-batch-size", "1", "--gradient-accumulation", str(gradient_accumulation),
        "--max-pixels", "200704", "--seed", str(seed), "--device", "cuda",
    ]
    print(json.dumps({"event": "training_command", "command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
    if not expected.exists():
        raise RuntimeError(f"fixed epoch-3 adapter missing after training: {expected}")
    return expected.parent


def recover_partial(path: Path, checkpoint_sha256: str) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                if row["checkpoint_sha256"] != checkpoint_sha256:
                    raise RuntimeError("partial predictions use a different checkpoint")
            except json.JSONDecodeError:
                break
            completed.add(row["query_id"])
            valid_end = handle.tell()
    if valid_end != path.stat().st_size:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
    return completed


def load_model(adapter: Path):
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL), local_files_only=True)
    processor.image_processor.max_pixels = 602112
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL), torch_dtype=torch.float16, local_files_only=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).to("cuda").eval()
    tokens = [
        processor.tokenizer.convert_tokens_to_ids("True"),
        processor.tokenizer.convert_tokens_to_ids("False"),
    ]
    return processor, model, tokens


@torch.inference_mode()
def score_batch(processor, model, token_ids: list[int], query: str, constraints: dict[str, Any], paths: list[Path]) -> list[float]:
    images = [Image.open(path).convert("RGB") for path in paths]
    rendered = constraint_text({"constraints": constraints}, True)
    texts = [
        processor.apply_chat_template(message(query, image, rendered), tokenize=False, add_generation_prompt=True)
        for image in images
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    # Match Qwen2-VL forward exactly while projecting only the final token. At
    # batch size 2 this was bitwise checked against the public full-logit path.
    base = model.get_base_model()
    input_ids = inputs["input_ids"]
    inputs_embeds = base.model.embed_tokens(input_ids)
    pixel_values = inputs["pixel_values"].type(base.visual.get_dtype())
    image_embeds = base.visual(pixel_values, grid_thw=inputs.get("image_grid_thw"))
    image_mask = (input_ids == base.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    )
    outputs = base.model(
        input_ids=None, attention_mask=inputs.get("attention_mask"),
        inputs_embeds=inputs_embeds, use_cache=False, return_dict=True,
    )
    logits = base.lm_head(outputs.last_hidden_state[:, -1, :]).float()
    return (logits[:, token_ids[0]] - logits[:, token_ids[1]]).cpu().tolist()


def infer_candidates(
    run_dir: Path,
    identifier: str,
    split: dict[str, Any],
    adapter: Path,
    checkpoint_sha256: str,
    training_manifest: dict[str, Any],
    physical_gpu_id: int,
    inference_batch_size: int,
) -> tuple[Path, dict[str, Any]]:
    language = split["language"]
    qids = split["validation_qids"] if split["kind"] == "inner" else split["test_qids"]
    cache = load_candidate_cache(run_dir, language)
    output_root = run_dir / ("inner_predictions" if split["kind"] == "inner" else "outer_predictions")
    output = output_root / f"{identifier}.jsonl"
    partial = output.with_suffix(".partial.jsonl")
    completed = recover_partial(partial, checkpoint_sha256)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    processor, model, token_ids = load_model(adapter)
    newly_scored = 0
    missing_images = []
    with partial.open("a", encoding="utf-8") as handle:
        for number, qid in enumerate(qids, 1):
            if qid in completed:
                continue
            source = cache[qid]
            candidates = union_candidates(source)
            valid_indices, paths = [], []
            for index, item in enumerate(candidates):
                image = IMAGE_ROOTS[language] / item["candidate_id"]
                item["image_path"] = str(image)
                item["image_exists"] = image.exists()
                item["finconrank_score"] = None
                if image.exists():
                    valid_indices.append(index)
                    paths.append(image)
                else:
                    missing_images.append({"query_id": qid, "candidate_id": item["candidate_id"], "path": str(image)})
            scores: list[float] = []
            query_started = time.perf_counter()
            for offset in range(0, len(paths), inference_batch_size):
                scores.extend(score_batch(
                    processor, model, token_ids, source["query"], source["constraints"],
                    paths[offset : offset + inference_batch_size],
                ))
            timestamp = utc_now()
            for index, score in zip(valid_indices, scores):
                candidates[index]["finconrank_score"] = float(score)
                candidates[index]["inference_timestamp"] = timestamp
                candidates[index]["provenance_status"] = "new_gpu_inference_current_strict_nested_checkpoint"
            newly_scored += len(scores)
            record = {
                "task_id": identifier, "language": language,
                "outer_fold": split["outer_fold"], "inner_fold": split.get("inner_fold"),
                "split": "inner_validation" if split["kind"] == "inner" else "outer_test_unscored",
                "query_id": qid, "source_report_group": source["group_id"], "query": source["query"],
                "query_type": source["query_type"], "checkpoint_path": str(adapter.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_training_report_groups_sha256": training_manifest["training_report_groups_sha256"],
                "checkpoint_training_report_group_count": training_manifest["training_report_group_count"],
                "physical_gpu_id": physical_gpu_id,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "inference_timestamp": timestamp,
                "query_inference_seconds": time.perf_counter() - query_started,
                "provenance_status": "passed_disjoint_query_groups_shared_corpus",
                "candidates": candidates,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed.add(qid)
            if number % 10 == 0 or len(completed) == len(qids):
                print(json.dumps({
                    "event": "inference_progress", "task_id": identifier,
                    "queries": len(completed), "expected": len(qids),
                    "newly_scored_candidates": newly_scored, "missing_images": len(missing_images),
                }), flush=True)
    if completed != set(qids):
        raise RuntimeError(f"inference incomplete: {len(completed)} != {len(qids)}")
    partial.replace(output)
    rows = read_jsonl(output)
    candidate_count = sum(len(row["candidates"]) for row in rows)
    scored_count = sum(candidate["finconrank_score"] is not None for row in rows for candidate in row["candidates"])
    stats = {
        "query_count": len(rows), "candidate_count": candidate_count,
        "scored_candidate_count": scored_count, "missing_image_count": len(missing_images),
        "coverage": scored_count / candidate_count if candidate_count else 0.0,
        "newly_scored_candidates_this_attempt": newly_scored,
        "inference_wall_seconds_this_attempt": time.perf_counter() - started,
        "inference_peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "prediction_path": str(output.resolve()), "prediction_sha256": sha256_file(output),
        "scoring_implementation": "equivalent_last_token_lm_head_batch2",
    }
    return output, stats


def provenance_check(split: dict[str, Any], training_manifest: dict[str, Any]) -> dict[str, Any]:
    training = set(training_manifest["training_report_groups"])
    if split["kind"] == "inner":
        inference = set(split["validation_report_groups"])
    else:
        inference = set(split["test_report_groups"])
    outer_test = set(split["outer_test_report_groups"] if split["kind"] == "inner" else split["test_report_groups"])
    overlap = sorted(training & inference)
    outer_overlap = sorted(training & outer_test)
    result = {
        "training_inference_report_group_overlap": overlap,
        "training_outer_test_report_group_overlap": outer_overlap,
        "passed": not overlap and not outer_overlap,
    }
    if not result["passed"]:
        raise RuntimeError(f"report-group provenance failure: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=2)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    identifier = args.task_id
    work = task_dir(run_dir, identifier)
    work.mkdir(parents=True, exist_ok=True)
    split = read_json(run_dir / f"splits/{identifier}.json")
    state_path = work / "state.json"
    started_at = utc_now()
    state = {
        "task_id": identifier, "kind": split["kind"], "language": split["language"],
        "outer_fold": split["outer_fold"], "inner_fold": split.get("inner_fold"),
        "attempt": args.attempt, "status": "running", "started_at": started_at,
        "pid": os.getpid(), "physical_gpu_id": args.physical_gpu_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
    }
    atomic_json(state_path, state)
    write_gpu_snapshot(work / f"attempt_{args.attempt}_nvidia_smi_before.txt")
    attempt_dir = work / f"attempt_{args.attempt}"
    training_dir = work / "training_data"
    try:
        training_manifest = build_training_pairs(run_dir, identifier, split, training_dir)
        provenance = provenance_check(split, training_manifest)
        atomic_json(work / "provenance_audit.json", provenance)
        checkpoint_manifest_path = work / "checkpoint_manifest.json"
        if checkpoint_manifest_path.exists():
            checkpoint_manifest = read_json(checkpoint_manifest_path)
            adapter = Path(checkpoint_manifest["adapter_path"])
            checkpoint_sha = checkpoint_manifest["adapter_sha256"]
            checkpoint_dir = adapter.parent
            if sha256_file(adapter / "adapter_model.safetensors") != checkpoint_sha:
                raise RuntimeError("preserved retry checkpoint hash mismatch")
            if checkpoint_manifest["training_report_groups_sha256"] != training_manifest["training_report_groups_sha256"]:
                raise RuntimeError("preserved retry checkpoint training-group mismatch")
            print(json.dumps({"event": "reuse_task_checkpoint_for_resumable_inference", "adapter": str(adapter)}), flush=True)
        else:
            checkpoint_dir = attempt_dir / "checkpoint"
            adapter = train_adapter(
                training_dir, checkpoint_dir, args.micro_batch_size,
                args.gradient_accumulation,
                SEED + split["outer_fold"] * 100 + (split.get("inner_fold") or 0) + (0 if split["language"] == "en" else 10000),
            )
            checkpoint_file = adapter / "adapter_model.safetensors"
            checkpoint_sha = sha256_file(checkpoint_file)
            atomic_json(checkpoint_manifest_path, {
                "task_id": identifier, "adapter_path": str(adapter.resolve()),
                "adapter_sha256": checkpoint_sha, "selected_epoch": 3,
                "selection_rule": "fixed protocol epoch; no validation label used",
                "training_report_groups_sha256": training_manifest["training_report_groups_sha256"],
                "physical_gpu_id": args.physical_gpu_id,
            })
        output, inference = infer_candidates(
            run_dir, identifier, split, adapter, checkpoint_sha, training_manifest,
            args.physical_gpu_id, args.inference_batch_size,
        )
        write_gpu_snapshot(work / f"attempt_{args.attempt}_nvidia_smi_after.txt")
        training_summary = read_json(checkpoint_dir / "training_summary.json")
        state.update({
            "status": "complete", "finished_at": utc_now(),
            "training": {
                "pairs": training_manifest["training_pair_count"],
                "queries": training_manifest["training_query_count"],
                "report_groups": training_manifest["training_report_group_count"],
                "seconds": training_summary["total_seconds"],
                "optimizer_steps": training_summary["global_optimizer_steps"],
                "peak_vram_gib": training_summary["max_vram_gib"],
            },
            "checkpoint_path": str(adapter.resolve()), "checkpoint_sha256": checkpoint_sha,
            "prediction_path": str(output.resolve()), "inference": inference,
            "provenance": provenance,
        })
        atomic_json(state_path, state)
        print(json.dumps({"event": "task_complete", **state}, ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update({"status": "failed", "failed_at": utc_now(), "error": repr(exc)})
        atomic_json(state_path, state)
        try:
            write_gpu_snapshot(work / f"attempt_{args.attempt}_nvidia_smi_failed.txt")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
