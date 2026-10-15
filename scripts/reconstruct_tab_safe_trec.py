#!/usr/bin/env python3
"""Reconstruct tab-safe TREC files from legacy whitespace TREC output.

The first local run wrote TREC fields separated by spaces while FinRAGBench-V
IDs may themselves contain spaces. This script uses the authoritative qids from
qrels to recover the qid boundary, then parses rank/score/run_name from the
right side of each line and writes official-style tab-separated TREC.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

import pandas as pd


def load_qids(qrels_path: Path, queries_path: Path | None = None) -> list[str]:
    frame = pd.read_csv(qrels_path, sep="\t")
    qids = {str(qid) for qid in frame["query-id"].tolist()}
    if queries_path is not None:
        queries = pd.read_parquet(queries_path)
        qids.update(str(qid) for qid in queries["query-id"].tolist())
    return sorted(qids, key=len, reverse=True)


def reconstruct_line(line: str, qids: list[str]) -> tuple[str, str, str, str, str]:
    for qid in qids:
        marker = f"{qid} Q0 "
        if line.startswith(marker):
            rest = line[len(marker) :]
            try:
                docid, rank, score, run_name = rest.rsplit(" ", 3)
            except ValueError as exc:
                raise ValueError(f"cannot parse TREC tail: {line}") from exc
            return qid, docid, rank, score, run_name
    raise ValueError(f"cannot match qid for line: {line}")


def reconstruct(input_path: Path, qrels_path: Path, output_path: Path, queries_path: Path | None = None) -> dict[str, int]:
    qids = load_qids(qrels_path, queries_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    recovered_qids: set[str] = set()
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            line = raw.rstrip("\n")
            if not line:
                continue
            qid, docid, rank, score, run_name = reconstruct_line(line, qids)
            recovered_qids.add(qid)
            line_count += 1
            dst.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{run_name}\n")
    return {
        "lines": line_count,
        "qids_in_qrels": len(qids),
        "qids_recovered": len(recovered_qids),
        "qids_missing": len(set(qids) - recovered_qids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = reconstruct(args.input, args.qrels, args.output, args.queries)
    print(stats)


if __name__ == "__main__":
    main()
