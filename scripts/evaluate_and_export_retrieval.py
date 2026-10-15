#!/usr/bin/env python3
"""Evaluate TREC retrieval output and export metrics/predictions JSONL."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Any

import pandas as pd


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(path, sep="\t")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in frame.to_dict("records"):
        qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return dict(qrels)


def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_parquet(path)
    return {str(row["query-id"]): row for row in frame.to_dict("records")}


def parse_trec(path: Path, *, fmt: str = "tab-safe") -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if fmt == "tab-safe":
            parts = line.split("\t")
            if len(parts) != 6:
                raise ValueError(f"bad tab-safe TREC line: {line}")
            qid, _, docid, rank, score, run_name = parts
        elif fmt == "legacy-whitespace":
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"bad legacy TREC line: {line}")
            qid = parts[0]
            docid = " ".join(parts[2:-3])
            rank, score, run_name = parts[-3:]
        else:
            raise ValueError(f"unknown TREC format: {fmt}")
        by_query[qid].append(
            {
                "corpus_id": docid,
                "rank": int(rank),
                "score": float(score),
                "run_name": run_name,
            }
        )
    for hits in by_query.values():
        hits.sort(key=lambda item: item["rank"])
    return dict(by_query)


def dcg(scores: list[int]) -> float:
    return sum((2**score - 1) / log2(index + 2) for index, score in enumerate(scores))


def ndcg_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    gains = [relevant.get(doc_id, 0) for doc_id in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(gains) / denom


def recall_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def mrr_at(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(run: dict[str, list[dict[str, Any]]], qrels: dict[str, dict[str, int]]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    per_query = {}
    rows = []
    for qid in sorted(qrels):
        relevant = {doc_id: score for doc_id, score in qrels[qid].items() if score > 0}
        ranked = [hit["corpus_id"] for hit in run.get(qid, [])]
        metrics = {
            "nDCG@5": ndcg_at(ranked, relevant, 5),
            "nDCG@10": ndcg_at(ranked, relevant, 10),
            "Recall@5": recall_at(ranked, relevant, 5),
            "Recall@10": recall_at(ranked, relevant, 10),
            "MRR@10": mrr_at(ranked, relevant, 10),
        }
        per_query[qid] = metrics
        rows.append(metrics)
    metrics = {
        "query_count": len(qrels),
        "nDCG@5": mean([row["nDCG@5"] for row in rows]),
        "nDCG@10": mean([row["nDCG@10"] for row in rows]),
        "Recall@5": mean([row["Recall@5"] for row in rows]),
        "Recall@10": mean([row["Recall@10"] for row in rows]),
        "MRR@10": mean([row["MRR@10"] for row in rows]),
    }
    return metrics, per_query


def export_predictions(
    out_path: Path,
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
    queries: dict[str, dict[str, Any]],
    per_query: dict[str, dict[str, float]],
) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(qrels):
            record = {
                "query_id": qid,
                "query": queries.get(qid, {}).get("query"),
                "answer": queries.get(qid, {}).get("answer"),
                "qrels": qrels[qid],
                "metrics": per_query[qid],
                "candidates": run.get(qid, []),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--trec-format", choices=["tab-safe", "legacy-whitespace"], default="tab-safe")
    args = parser.parse_args()

    run = parse_trec(args.trec, fmt=args.trec_format)
    qrels = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    metrics, per_query = evaluate(run, qrels)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    export_predictions(args.predictions_out, run, qrels, queries, per_query)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
