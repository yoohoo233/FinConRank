#!/usr/bin/env python3
"""Locked small-sample deployment-cost benchmark for the major revision."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import gc
import hashlib
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

from finconrank.strict_nested import (
    IMAGE_ROOTS, LANGUAGES, atomic_csv, atomic_json, atomic_jsonl,
    read_json, read_jsonl, sha256_file, stable_hash, task_id, utc_now,
)
from run_major_revision_docrerank import (
    MAX_PIXELS as DOC_MAX_PIXELS, MIN_PIXELS as DOC_MIN_PIXELS,
    PROMPT_TEMPLATE, official_messages, score_probability,
)
from train_finconrank import constraint_text, message


SEED = 42
DEPTH = 20
BATCH_SIZE = 2
MONO_MAX_PIXELS = 602112


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for item in items:
        cid = str(item["corpus_id"])
        if cid not in seen:
            seen.add(cid)
            output.append(dict(item))
    return output


def prepare(strict_run: Path, revision_run: Path, source_run: Path, mono_adapter: Path, doc_adapter: Path) -> None:
    output_dir = revision_run / "results" / "cost_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = read_jsonl(source_run / "query_constraints.jsonl")
    strict_rows = {}
    for language in LANGUAGES:
        strict_rows.update({
            (language, row["query_id"]): row
            for row in read_jsonl(strict_run / "frozen_inputs" / f"{language}_candidates_without_qrels.jsonl")
        })
    outer_lookup = {}
    for language in LANGUAGES:
        for outer in range(5):
            split = read_json(strict_run / "splits" / f"outer_{language}_fold{outer}.json")
            for qid in split["test_qids"]:
                outer_lookup[(language, qid)] = outer
    selected = []
    for language in LANGUAGES:
        eligible = [
            row for row in split_rows
            if row["language"] == language and row["split"] == "dev" and (language, row["query_id"]) in strict_rows
        ]
        groups = defaultdict(list)
        for row in eligible:
            groups[row["group_id"]].append(row)
        chosen_groups = sorted(groups, key=lambda group: stable_key(f"{language}:group:{group}"))[:20]
        for group in chosen_groups:
            row = sorted(groups[group], key=lambda item: stable_key(f"{language}:query:{item['query_id']}"))[0]
            source = strict_rows[(language, row["query_id"])]
            outer = outer_lookup[(language, row["query_id"])]
            state = read_json(strict_run / "tasks" / task_id("outer", language, outer) / "state.json")
            candidates = deduplicate(source["monoqwen2vl_candidates"][:DEPTH])
            selected.append({
                "language": language, "query_id": row["query_id"], "source_report_group": group,
                "outer_fold": outer, "query": row["query"], "constraints": row["constraints"],
                "candidate_policy": "first-occurrence-deduplicated frozen MonoQwen2-VL top-20",
                "candidates": [
                    {"candidate_id": item["corpus_id"], "image_path": str((IMAGE_ROOTS[language] / item["corpus_id"]).resolve())}
                    for item in candidates
                ],
                "finconrank_adapter": state["checkpoint_path"],
                "finconrank_adapter_sha256": state["checkpoint_sha256"],
            })
    sample_path = output_dir / "locked_timing_queries.jsonl"
    atomic_jsonl(sample_path, selected)
    protocol = {
        "status": "locked_before_cost_gpu_measurement", "locked_at": utc_now(), "seed": SEED,
        "selection": "up to 20 deterministic report groups per language from the original development split; one deterministic query per group; no effectiveness-based sampling",
        "query_counts": {language: sum(row["language"] == language for row in selected) for language in LANGUAGES},
        "candidate_depth": DEPTH, "batch_size": BATCH_SIZE, "warmup_queries": 1,
        "routes": {
            "monoqwen2vl": "base plus frozen MonoQwen2-VL adapter, fp16, max_pixels=602112",
            "finconrank": "base plus query's strict outer-fold FinConRank adapter, fp16, max_pixels=602112; adapter loads reported separately",
            "docrerank": "base plus fixed DocReRank adapter, bf16, min_pixels=200704, max_pixels=589824",
            "three_score": "cached ColQ + newly timed Mono + cached Metric-Aware CPU fusion",
            "complete_system": "cached ColQ + newly timed Mono and FinConRank + cached Metric-Aware CPU fusion",
        },
        "base_model": read_json(strict_run / "protocol_lock.json")["model_recipe"]["base_model"],
        "mono_adapter": str(mono_adapter), "doc_adapter": str(doc_adapter),
        "sample_path": str(sample_path), "sample_sha256": sha256_file(sample_path),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
        "constraints": "one model at a time on one uncontended physical RTX 4090; CUDA synchronized; model load, image decode, preprocess, GPU inference, and CPU fusion separated",
    }
    protocol["protocol_sha256"] = stable_hash(protocol)
    atomic_json(output_dir / "protocol_lock.json", protocol)
    print(json.dumps({"status": "prepared", "queries": len(selected), "pairs": sum(len(row["candidates"]) for row in selected)}, indent=2))


def verify(output_dir: Path) -> dict[str, Any]:
    protocol = read_json(output_dir / "protocol_lock.json")
    if protocol.get("status") != "locked_before_cost_gpu_measurement":
        raise RuntimeError("cost protocol is not locked")
    if sha256_file(Path(protocol["sample_path"])) != protocol["sample_sha256"]:
        raise RuntimeError("timing query list changed")
    addendum_path = output_dir / "implementation_addendum.json"
    expected_script = read_json(addendum_path)["script_sha256"] if addendum_path.exists() else protocol["script_sha256"]
    if sha256_file(Path(__file__).resolve()) != expected_script:
        raise RuntimeError("cost benchmark script changed")
    return protocol


def unload(model: Any, processor: Any) -> None:
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_adapter(base_path: str, adapter: str, dtype: torch.dtype, min_pixels: int | None, max_pixels: int) -> tuple[Any, Any, list[int], float]:
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(base_path, local_files_only=True)
    if min_pixels is not None:
        processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to("cuda:0").eval()
    token_ids = [processor.tokenizer.convert_tokens_to_ids("True"), processor.tokenizer.convert_tokens_to_ids("False")]
    torch.cuda.synchronize()
    return processor, model, token_ids, time.perf_counter() - started


@torch.inference_mode()
def score_query(processor: Any, model: Any, tokens: list[int], row: dict[str, Any], route: str) -> dict[str, float]:
    totals = defaultdict(float)
    paths = [Path(item["image_path"]) for item in row["candidates"]]
    for offset in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[offset:offset + BATCH_SIZE]
        started = time.perf_counter()
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        totals["image_decode_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        if route == "finconrank":
            rendered = constraint_text({"constraints": row["constraints"]}, True)
            texts = [processor.apply_chat_template(message(row["query"], image, rendered), tokenize=False, add_generation_prompt=True) for image in images]
        else:
            texts = [processor.apply_chat_template(official_messages(row["query"], image), tokenize=False, add_generation_prompt=True) for image in images]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        totals["preprocess_seconds"] += time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(**inputs).logits[:, -1, :]
        if route == "finconrank":
            _ = (logits[:, tokens[0]].float() - logits[:, tokens[1]].float()).cpu()
        else:
            _ = score_probability(logits, tokens[0], tokens[1]).cpu()
        torch.cuda.synchronize()
        totals["gpu_inference_seconds"] += time.perf_counter() - started
    totals["candidate_count"] = len(paths)
    totals["query_active_seconds"] = totals["image_decode_seconds"] + totals["preprocess_seconds"] + totals["gpu_inference_seconds"]
    return dict(totals)


def benchmark(revision_run: Path, route: str) -> None:
    output_dir = revision_run / "results" / "cost_benchmark"
    protocol = verify(output_dir)
    rows = read_jsonl(Path(protocol["sample_path"]))
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    wall_started = time.perf_counter()
    load_total = 0.0
    if route in ("monoqwen2vl", "docrerank"):
        adapter = protocol["mono_adapter"] if route == "monoqwen2vl" else protocol["doc_adapter"]
        dtype = torch.float16 if route == "monoqwen2vl" else torch.bfloat16
        min_pixels = None if route == "monoqwen2vl" else DOC_MIN_PIXELS
        max_pixels = MONO_MAX_PIXELS if route == "monoqwen2vl" else DOC_MAX_PIXELS
        processor, model, tokens, load_seconds = load_adapter(protocol["base_model"], adapter, dtype, min_pixels, max_pixels)
        load_total += load_seconds
        for index, row in enumerate(rows):
            values = score_query(processor, model, tokens, row, route)
            measurements.append({
                "route": route, "language": row["language"], "query_id": row["query_id"],
                "source_report_group": row["source_report_group"], "outer_fold": row["outer_fold"],
                "warmup": index == 0, **values,
            })
        unload(model, processor)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["finconrank_adapter"]].append(row)
        for adapter, adapter_rows in sorted(grouped.items()):
            processor, model, tokens, load_seconds = load_adapter(protocol["base_model"], adapter, torch.float16, None, MONO_MAX_PIXELS)
            load_total += load_seconds
            for index, row in enumerate(adapter_rows):
                values = score_query(processor, model, tokens, row, route)
                measurements.append({
                    "route": route, "language": row["language"], "query_id": row["query_id"],
                    "source_report_group": row["source_report_group"], "outer_fold": row["outer_fold"],
                    "adapter": adapter, "warmup": index == 0, **values,
                })
            unload(model, processor)
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
    output = output_dir / f"{route}_query_timing.csv"
    atomic_csv(output, measurements)
    summary = {
        "route": route, "physical_gpu_id": int(os.environ.get("FINCONRANK_PHYSICAL_GPU_ID", "-1")),
        "gpu_name": torch.cuda.get_device_name(0), "query_count": len(measurements),
        "candidate_count": int(sum(row["candidate_count"] for row in measurements)),
        "model_load_seconds": load_total, "wall_seconds": time.perf_counter() - wall_started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "output": str(output), "output_sha256": sha256_file(output), "status": "complete", "finished_at": utc_now(),
    }
    atomic_json(output_dir / f"{route}_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--strict-run", type=Path, required=True)
    prep.add_argument("--revision-run", type=Path, required=True)
    prep.add_argument("--source-run", type=Path, required=True)
    prep.add_argument("--mono-adapter", type=Path, required=True)
    prep.add_argument("--doc-adapter", type=Path, required=True)
    work = sub.add_parser("benchmark")
    work.add_argument("--revision-run", type=Path, required=True)
    work.add_argument("--route", choices=("monoqwen2vl", "finconrank", "docrerank"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.strict_run.resolve(), args.revision_run.resolve(), args.source_run.resolve(), args.mono_adapter.resolve(), args.doc_adapter.resolve())
    else:
        benchmark(args.revision_run.resolve(), args.route)


if __name__ == "__main__":
    main()
