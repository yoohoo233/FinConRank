#!/usr/bin/env python3
"""Score official ColQwen2 embeddings and export a TREC run.

This keeps the official ColQwen2 late-interaction/MaxSim scoring formula, but
decouples scoring from the slow single-process result aggregation in the
reference retrieval entry point.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import heapq
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import torch


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_queries(run_dir: Path, rank: int = 0):
    embeddings = []
    lookup = []
    for path in sorted(run_dir.glob(f"embeddings.query.rank.{rank}*")):
        data = load_pickle(path)
        if len(data[1]) == 0:
            continue
        embeddings.extend(data[0])
        lookup.extend(data[1])
    if not embeddings:
        raise ValueError(f"No query embeddings found in {run_dir}")
    return embeddings, lookup


def score_multi_vector(
    qs: list[torch.Tensor],
    ps: list[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    q_batches = [
        torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size],
            batch_first=True,
            padding_value=0,
        ).to(device)
        for i in range(0, len(qs), batch_size)
    ]
    p_batches = [
        torch.nn.utils.rnn.pad_sequence(
            ps[j : j + batch_size],
            batch_first=True,
            padding_value=0,
        ).to(device)
        for j in range(0, len(ps), batch_size)
    ]
    scores_list: list[torch.Tensor] = []
    for qs_batch in q_batches:
        scores_batch: list[torch.Tensor] = []
        for ps_batch in p_batches:
            scores_batch.append(
                torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                .max(dim=3)[0]
                .sum(dim=2)
                .detach()
                .cpu()
            )
        scores_list.append(torch.cat(scores_batch, dim=1))
        del scores_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores = torch.cat(scores_list, dim=0).to(torch.float32)
    del q_batches, p_batches
    return scores


def trim_results(results: dict[str, list[tuple[str, float]]], topk: int, factor: int = 8) -> None:
    limit = topk * factor
    for qid, items in list(results.items()):
        if len(items) > limit:
            results[qid] = heapq.nlargest(topk, items, key=lambda item: item[1])


def write_partial(path: Path, results: dict[str, list[tuple[str, float]]], topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(results):
            ranked = heapq.nlargest(topk, results[qid], key=lambda item: item[1])
            f.write(json.dumps({"qid": qid, "hits": ranked}, ensure_ascii=False) + "\n")


def read_partial(path: Path) -> Iterable[tuple[str, list[tuple[str, float]]]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["qid"], [(docid, float(score)) for docid, score in row["hits"]]


def write_trec(path: Path, results: dict[str, list[tuple[str, float]]], topk: int, run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(results):
            ranked = heapq.nlargest(topk, results[qid], key=lambda item: item[1])
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.8f}\t{run_name}\n")


def score_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    queries, query_ids = load_queries(args.run_dir, args.query_rank)
    corpus_parts = sorted(args.run_dir.glob(args.corpus_glob))
    assigned_parts = [
        path for index, path in enumerate(corpus_parts) if index % args.num_workers == args.worker_index
    ]
    if not assigned_parts:
        raise ValueError("No corpus shards assigned to this worker")

    results: dict[str, list[tuple[str, float]]] = {qid: [] for qid in query_ids}
    for shard_index, path in enumerate(assigned_parts, start=1):
        corpus_embeddings, corpus_ids = load_pickle(path)
        if len(corpus_ids) == 0:
            continue
        scores = score_multi_vector(
            queries,
            corpus_embeddings,
            batch_size=args.score_batch_size,
            device=device,
        )
        shard_topk = min(args.topk, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, shard_topk, dim=1, largest=True, sorted=False)
        for qi, qid in enumerate(query_ids):
            results[qid].extend(
                (corpus_ids[int(idx)], float(score))
                for idx, score in zip(top_indices[qi].tolist(), top_scores[qi].tolist())
            )
        if shard_index % args.trim_every == 0:
            trim_results(results, args.topk)
        print(
            json.dumps(
                {
                    "worker": args.worker_index,
                    "done": shard_index,
                    "total": len(assigned_parts),
                    "shard": path.name,
                }
            ),
            flush=True,
        )
        del corpus_embeddings, corpus_ids, scores, top_scores, top_indices
        if device.type == "cuda":
            torch.cuda.empty_cache()

    trim_results(results, args.topk, factor=1)
    write_partial(args.partial_out, results, args.topk)


def merge_partials(args: argparse.Namespace) -> None:
    results: dict[str, list[tuple[str, float]]] = {}
    for partial in args.partials:
        for qid, hits in read_partial(partial):
            results.setdefault(qid, []).extend(hits)
    trim_results(results, args.topk, factor=1)
    write_trec(args.trec_out, results, args.topk, args.run_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-worker")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--device", required=True)
    score.add_argument("--worker-index", type=int, required=True)
    score.add_argument("--num-workers", type=int, required=True)
    score.add_argument("--partial-out", type=Path, required=True)
    score.add_argument("--topk", type=int, default=100)
    score.add_argument("--query-rank", type=int, default=0)
    score.add_argument("--corpus-glob", default="embeddings.corpus.rank.*")
    score.add_argument("--score-batch-size", type=int, default=16)
    score.add_argument("--trim-every", type=int, default=4)
    score.set_defaults(func=score_worker)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--partials", type=Path, nargs="+", required=True)
    merge.add_argument("--trec-out", type=Path, required=True)
    merge.add_argument("--topk", type=int, default=100)
    merge.add_argument("--run-name", default="colqwen2")
    merge.set_defaults(func=merge_partials)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "num_workers") and args.worker_index >= args.num_workers:
        raise ValueError("--worker-index must be smaller than --num-workers")
    if hasattr(args, "score_batch_size") and args.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
