#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from finconrank.common import read_jsonl


BASE_PROMPT = (
    "Judge whether the financial document page is relevant to the query. "
    "Pay special attention to company/entity, metric, year/period, unit/currency, "
    "statement type, and accounting scope. Answer True or False. Query: {query}{constraints}"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def constraint_text(item: dict[str, Any], enabled: bool) -> str:
    if not enabled:
        return ""
    c = item.get("constraints", {})
    values = {
        "entity": c.get("entity"),
        "metric": c.get("canonical_metrics", []),
        "year": c.get("year_terms", []),
        "period": c.get("period_terms", []),
        "unit": c.get("unit_terms", []),
        "statement": c.get("statement_type"),
        "scope": c.get("accounting_scope", []),
    }
    rendered = "; ".join(f"{key}={value}" for key, value in values.items() if value and value != "unknown")
    return f"\nParsed financial constraints: {rendered}." if rendered else ""


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def filter_variant(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant != "generic_hard_negatives":
        return rows
    return [row for row in rows if int(row.get("curriculum_stage", 2)) < 2]


def stage_rows(rows: list[dict[str, Any]], epoch: int, curriculum: bool) -> list[dict[str, Any]]:
    if not curriculum:
        return rows
    allowed = set(range(min(epoch, 2) + 1))
    return [row for row in rows if int(row.get("curriculum_stage", 2)) in allowed]


def message(query: str, image: Image.Image, constraints: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": BASE_PROMPT.format(query=query, constraints=constraints)}]}]


def prepare_batch(processor, rows: list[dict[str, Any]], use_constraints: bool, max_pixels: int, device: str):
    images = []
    texts = []
    for row in rows:
        constraints = constraint_text(row, use_constraints)
        for key in ("positive_image", "negative_image"):
            image = Image.open(row[key]).convert("RGB")
            images.append(image)
            texts.append(processor.apply_chat_template(message(row["query"], image, constraints), tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return {key: value.to(device) for key, value in inputs.items()}


def pair_scores(model, inputs: dict[str, torch.Tensor], token_ids: list[int], pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    relevance = logits[:, token_ids[0]] - logits[:, token_ids[1]]
    return relevance[0::2][:pair_count], relevance[1::2][:pair_count]


@torch.no_grad()
def evaluate_pairwise(model, processor, rows: list[dict[str, Any]], token_ids: list[int], args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    margins = []
    type_stats: dict[str, list[int]] = {}
    loader = DataLoader(PairDataset(rows), batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda batch: batch)
    for batch in loader:
        inputs = prepare_batch(processor, batch, args.use_constraints, args.max_pixels, args.device)
        positive, negative = pair_scores(model, inputs, token_ids, len(batch))
        diffs = (positive - negative).float().cpu().tolist()
        for row, diff in zip(batch, diffs):
            hit = int(diff > 0)
            correct += hit
            total += 1
            margins.append(diff)
            for kind in row.get("negative_types", []):
                type_stats.setdefault(kind, [0, 0])
                type_stats[kind][0] += hit
                type_stats[kind][1] += 1
        del inputs, positive, negative
    return {
        "pairwise_accuracy": correct / total if total else 0.0,
        "pair_count": total,
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "by_negative_type": {key: {"accuracy": value[0] / value[1], "count": value[1]} for key, value in sorted(type_stats.items())},
    }


def save_adapter(model, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--dev-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--init-adapter", default="lightonai/MonoQwen2-VL-v0.1")
    parser.add_argument("--variant", choices=["full", "generic_hard_negatives", "no_curriculum", "no_constraints", "no_false_negative_verification"], default="full")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-dev-pairs", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.use_constraints = args.variant != "no_constraints"
    curriculum = args.variant not in {"no_curriculum", "generic_hard_negatives"}
    seed_everything(args.seed)

    train_rows = filter_variant(read_jsonl(args.train_pairs), args.variant)
    dev_rows = filter_variant(read_jsonl(args.dev_pairs), args.variant)
    if args.max_train_pairs:
        train_rows = train_rows[: args.max_train_pairs]
    if args.max_dev_pairs:
        dev_rows = dev_rows[: args.max_dev_pairs]

    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.init_adapter, local_files_only=True, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(args.device)
    true_id = processor.tokenizer.convert_tokens_to_ids("True")
    false_id = processor.tokenizer.convert_tokens_to_ids("False")
    token_ids = [true_id, false_id]
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "base_model": args.base_model,
        "init_adapter": args.init_adapter,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_pixels": args.max_pixels,
        "curriculum": curriculum,
        "use_constraints": args.use_constraints,
        "train_pairs": len(train_rows),
        "dev_pairs": len(dev_rows),
        "trainable_parameters": sum(param.numel() for param in trainable),
        "total_parameters": sum(param.numel() for param in model.parameters()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    history = []
    best_accuracy = -1.0
    global_step = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        active = stage_rows(train_rows, epoch, curriculum)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(PairDataset(active), batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=lambda batch: batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loader, start=1):
            inputs = prepare_batch(processor, batch, args.use_constraints, args.max_pixels, args.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                positive, negative = pair_scores(model, inputs, token_ids, len(batch))
                loss = F.margin_ranking_loss(positive, negative, torch.ones_like(positive), margin=args.margin)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if step % args.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu()))
            del inputs, positive, negative, loss
        dev = evaluate_pairwise(model, processor, dev_rows, token_ids, args)
        epoch_record = {
            "epoch": epoch + 1,
            "active_curriculum_stages": sorted({int(row.get("curriculum_stage", 2)) for row in active}),
            "active_train_pairs": len(active),
            "mean_train_loss": sum(losses) / len(losses),
            "optimizer_steps": global_step,
            "dev": dev,
            "elapsed_seconds": time.perf_counter() - started,
            "max_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if dev["pairwise_accuracy"] > best_accuracy:
            best_accuracy = dev["pairwise_accuracy"]
            best = args.output_dir / "best_adapter"
            save_adapter(model, best)
            (args.output_dir / "best_epoch.json").write_text(json.dumps(epoch_record, ensure_ascii=False, indent=2), encoding="utf-8")
        save_adapter(model, args.output_dir / f"epoch_{epoch + 1}_adapter")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    final = {
        **metadata,
        "best_dev_pairwise_accuracy": best_accuracy,
        "total_seconds": time.perf_counter() - started,
        "global_optimizer_steps": global_step,
        "max_vram_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
