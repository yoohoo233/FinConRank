#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from finconrank.common import read_jsonl, write_jsonl
from train_finconrank import BASE_PROMPT, constraint_text, message


def load_model(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).to(args.device).eval()
    return processor, model, [processor.tokenizer.convert_tokens_to_ids("True"), processor.tokenizer.convert_tokens_to_ids("False")]


@torch.no_grad()
def score_batch(processor, model, token_ids: list[int], query: str, constraints: dict[str, Any], pairs: list[tuple[str, Path]], args: argparse.Namespace) -> list[float]:
    images = [Image.open(path).convert("RGB") for _, path in pairs]
    ctext = constraint_text({"constraints": constraints}, args.use_constraints)
    texts = [processor.apply_chat_template(message(query, image, ctext), tokenize=False, add_generation_prompt=True) for image in images]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    logits = model(**inputs).logits[:, -1, :]
    scores = (logits[:, token_ids[0]] - logits[:, token_ids[1]]).float().cpu().tolist()
    return [float(score) for score in scores]


def worker(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    if args.qid_file:
        allowed = set(json.loads(args.qid_file.read_text(encoding="utf-8")))
        rows = [row for row in rows if row["query_id"] in allowed]
    constraints = {row["query_id"]: row for row in read_jsonl(args.constraints)}
    assigned = [row for index, row in enumerate(rows) if index % args.num_workers == args.worker_index]
    processor, model, token_ids = load_model(args)
    output = []
    total_pairs = 0
    total_seconds = 0.0
    missing = []
    for index, row in enumerate(assigned, start=1):
        started = time.perf_counter()
        info = constraints[row["query_id"]]
        candidates = row.get("candidates", [])[: args.candidate_depth]
        pairs = []
        kept = []
        for candidate in candidates:
            path = args.image_root / candidate["corpus_id"]
            item = dict(candidate)
            if not path.exists():
                item["finconrank_score"] = None
                item["finconrank_error"] = "missing image"
                missing.append({"query_id": row["query_id"], "corpus_id": candidate["corpus_id"], "path": str(path)})
            else:
                pairs.append((row["query"], path))
            kept.append(item)
        scores = []
        for offset in range(0, len(pairs), args.batch_size):
            scores.extend(score_batch(processor, model, token_ids, row["query"], info["constraints"], pairs[offset : offset + args.batch_size], args))
        score_iter = iter(scores)
        for item in kept:
            if "finconrank_score" not in item:
                item["finconrank_score"] = next(score_iter)
        elapsed = time.perf_counter() - started
        total_pairs += len(scores)
        total_seconds += elapsed
        output.append({
            **{key: value for key, value in row.items() if key != "candidates"},
            "language": args.language,
            "split": info["split"],
            "group_id": info["group_id"],
            "query_type": info["query_type"],
            "query_constraints": info["constraints"],
            "finconrank_checkpoint": str(args.adapter),
            "finconrank_query_seconds": elapsed,
            "candidates": kept,
        })
        if index % 20 == 0 or index == len(assigned):
            print(json.dumps({"worker": args.worker_index, "queries": index, "total": len(assigned), "pairs": total_pairs}), flush=True)
    write_jsonl(args.output, output)
    stats = {
        "worker": args.worker_index,
        "queries": len(output),
        "scored_pairs": total_pairs,
        "missing_images": missing,
        "total_seconds": total_seconds,
        "seconds_per_pair": total_seconds / total_pairs if total_pairs else 0.0,
        "max_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def merge(args: argparse.Namespace) -> None:
    order = [row["query_id"] for row in read_jsonl(args.original_predictions)]
    rows = {}
    for path in args.inputs:
        for row in read_jsonl(path):
            rows[row["query_id"]] = row
    missing_qids = [qid for qid in order if qid not in rows]
    if missing_qids:
        raise RuntimeError(f"missing {len(missing_qids)} query ids during merge")
    write_jsonl(args.output, [rows[qid] for qid in order])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker")
    work.add_argument("--predictions", type=Path, required=True)
    work.add_argument("--constraints", type=Path, required=True)
    work.add_argument("--image-root", type=Path, required=True)
    work.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    work.add_argument("--adapter", type=Path, required=True)
    work.add_argument("--language", choices=["en", "zh"], required=True)
    work.add_argument("--candidate-depth", type=int, default=30)
    work.add_argument("--qid-file", type=Path)
    work.add_argument("--batch-size", type=int, default=2)
    work.add_argument("--max-pixels", type=int, default=200704)
    work.add_argument("--device", default="cuda")
    work.add_argument("--worker-index", type=int, required=True)
    work.add_argument("--num-workers", type=int, required=True)
    work.add_argument("--use-constraints", action=argparse.BooleanOptionalAction, default=True)
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--stats", type=Path, required=True)
    work.set_defaults(func=worker)
    combine = sub.add_parser("merge")
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--original-predictions", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(func=merge)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "stats"):
        args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)
