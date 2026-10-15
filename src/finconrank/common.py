from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = ["nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def page_document_id(corpus_id: str) -> str:
    name = Path(corpus_id).name
    # Some derived ids end in `.png-s1`; Path.stem does not recognize that as
    # an image suffix. Normalize the image extension and optional crop suffix.
    stem = re.sub(r"\.(?:png|jpe?g)(?:-[A-Za-z]\d+)?$", "", name, flags=re.IGNORECASE)
    # FinRAGBench-V also contains stitched multipage images whose filename
    # retains the source-page range. The corpus contains both `multipage` and
    # a `multipgae` typo, plus derived names such as `.png-1_78.png`.
    multipage = re.match(
        r"^(.*)_multi(?:page|pgae)_[0-9]+(?:-[0-9]+)+(?:\.png-\d+)?(?:_[0-9]+)?$",
        stem,
        flags=re.IGNORECASE,
    )
    if multipage:
        return multipage.group(1)
    match = re.match(r"^(.*)_([0-9]+)$", stem)
    return match.group(1) if match else stem


def evaluate(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> dict[str, float]:
    rels = [1 if int(qrels.get(row["corpus_id"], 0)) > 0 else 0 for row in candidates]
    total_rel = max(1, sum(int(value) > 0 for value in qrels.values()))
    out: dict[str, float] = {}
    for k in (5, 10):
        observed = rels[:k]
        dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(observed))
        ideal = [1] * min(total_rel, k)
        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
        out[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
        out[f"Recall@{k}"] = sum(observed) / total_rel
    out["MRR@10"] = next((1.0 / (idx + 1) for idx, rel in enumerate(rels[:10]) if rel), 0.0)
    return out


def aggregate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    count = 0
    for row in rows:
        count += 1
        metrics = row.get("metrics") or evaluate(row.get("candidates", []), row.get("qrels", {}))
        for metric in METRICS:
            values[metric].append(float(metrics[metric]))
    return {metric: sum(values[metric]) / count if count else 0.0 for metric in METRICS}


def first_gold_rank(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> int | None:
    for rank, row in enumerate(candidates, start=1):
        if int(qrels.get(row["corpus_id"], 0)) > 0:
            return rank
    return None


def normalize_scores(candidates: list[dict[str, Any]], field: str) -> dict[str, float]:
    raw = {row["corpus_id"]: float(row.get(field, 0.0) or 0.0) for row in candidates}
    if not raw:
        return {}
    low, high = min(raw.values()), max(raw.values())
    if high <= low:
        return {key: 0.5 for key in raw}
    return {key: (value - low) / (high - low) for key, value in raw.items()}
