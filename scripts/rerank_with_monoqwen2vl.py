#!/usr/bin/env python3
"""Rerank ColQwen2 candidates with local MonoQwen2-VL."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


PROMPT = (
    "Assert the relevance of the previous image document to the following query, "
    "answer True or False. The query is: {query}"
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(query: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT.format(query=query)},
            ],
        }
    ]


def load_model(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).eval()
    true_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_id = processor.tokenizer.convert_tokens_to_ids("False")
    return processor, model, [true_id, false_id]


def score_batch(
    processor,
    model,
    token_ids: list[int],
    pairs: list[tuple[str, Path]],
    device: str,
) -> list[float]:
    images = [Image.open(path).convert("RGB") for _, path in pairs]
    texts = [
        processor.apply_chat_template(build_messages(query, image), tokenize=False, add_generation_prompt=True)
        for (query, _), image in zip(pairs, images)
    ]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    probs = torch.softmax(logits[:, token_ids], dim=-1)[:, 0].detach().cpu().tolist()
    del inputs, logits
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [float(score) for score in probs]


def worker(args: argparse.Namespace) -> None:
    rows = load_rows(args.predictions)
    assigned = [row for index, row in enumerate(rows) if index % args.num_workers == args.worker_index]
    processor, model, token_ids = load_model(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row_index, row in enumerate(assigned, start=1):
            candidates = row["candidates"][: args.top_n]
            pairs: list[tuple[str, Path]] = []
            kept: list[dict[str, Any]] = []
            for candidate in candidates:
                image_path = args.image_root / candidate["corpus_id"]
                if not image_path.exists():
                    candidate = dict(candidate)
                    candidate["rerank_score"] = None
                    candidate["rerank_error"] = "missing image"
                    candidate["rerank_missing_image_path"] = str(image_path)
                    candidate["rerank_query_id"] = row["query_id"]
                    kept.append(candidate)
                    continue
                pairs.append((row["query"], image_path))
                kept.append(dict(candidate))

            scores: list[float] = []
            for offset in range(0, len(pairs), args.batch_size):
                scores.extend(score_batch(processor, model, token_ids, pairs[offset : offset + args.batch_size], args.device))

            score_iter = iter(scores)
            for candidate in kept:
                if candidate.get("rerank_score") is None and "rerank_error" not in candidate:
                    candidate["rerank_score"] = next(score_iter)

            ranked = sorted(
                kept,
                key=lambda item: (
                    -1.0 if item.get("rerank_score") is None else float(item["rerank_score"]),
                    float(item.get("score", 0.0)),
                ),
                reverse=True,
            )
            for rank, candidate in enumerate(ranked, start=1):
                candidate["rank"] = rank
                candidate["run_name"] = args.run_name
            row["candidates"] = ranked[: args.output_depth]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps({"worker": args.worker_index, "done": row_index, "total": len(assigned)}),
                flush=True,
            )


def merge(args: argparse.Namespace) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for path in args.inputs:
        for row in load_rows(path):
            rows[row["query_id"]] = row
    ordered = load_rows(args.original_predictions)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.trec_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8") as pred_out, args.trec_out.open("w", encoding="utf-8") as trec_out:
        for original in ordered:
            row = rows[original["query_id"]]
            pred_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            for candidate in row["candidates"]:
                score = candidate.get("rerank_score")
                if score is None:
                    score = candidate.get("score", 0.0)
                trec_out.write(
                    f"{row['query_id']}\tQ0\t{candidate['corpus_id']}\t{candidate['rank']}\t{float(score):.8f}\t{candidate['run_name']}\n"
                )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker")
    work.add_argument("--predictions", type=Path, required=True)
    work.add_argument("--image-root", type=Path, required=True)
    work.add_argument("--base-model", required=True)
    work.add_argument("--adapter", required=True)
    work.add_argument("--device", default="cuda")
    work.add_argument("--worker-index", type=int, required=True)
    work.add_argument("--num-workers", type=int, required=True)
    work.add_argument("--batch-size", type=int, default=2)
    work.add_argument("--max-pixels", type=int, default=602112)
    work.add_argument("--top-n", type=int, default=20)
    work.add_argument("--output-depth", type=int, default=20)
    work.add_argument("--run-name", default="colqwen2_monoqwen2vl_rerank")
    work.add_argument("--output", type=Path, required=True)
    work.set_defaults(func=worker)

    merge_cmd = sub.add_parser("merge")
    merge_cmd.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_cmd.add_argument("--original-predictions", type=Path, required=True)
    merge_cmd.add_argument("--predictions-out", type=Path, required=True)
    merge_cmd.add_argument("--trec-out", type=Path, required=True)
    merge_cmd.set_defaults(func=merge)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
