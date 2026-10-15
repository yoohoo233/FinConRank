#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

from finconrank.release_paths import release_path

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finconrank.common import METRICS, first_gold_rank, page_document_id, read_jsonl, write_jsonl
from finconrank.constraints import METRICS as FIN_METRICS
from finconrank.constraints import STATEMENTS, UNITS, YEAR_RE, classify_query, extract_constraints, find_terms


DEPTHS = [10, 20, 30, 50, 100]


def select_page_texts(path: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cid = row["corpus_id"]
            if cid in wanted:
                found[cid] = f"{row.get('title', '')} {row.get('text', '')}".lower()
    return found


def connected_query_groups(rows: list[dict[str, Any]], language: str) -> dict[str, str]:
    parent: dict[str, str] = {}

    def root(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = root(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    q_docs: dict[str, list[str]] = {}
    for row in rows:
        docs = sorted({f"{language}:{page_document_id(cid)}" for cid, rel in row.get("qrels", {}).items() if int(rel) > 0})
        if not docs:
            docs = [f"{language}:ungrouped:{row['query_id']}"]
        q_docs[row["query_id"]] = docs
        for doc in docs[1:]:
            union(docs[0], doc)
    return {qid: root(docs[0]) for qid, docs in q_docs.items()}


def assign_group_split(rows_by_language: dict[str, list[dict[str, Any]]], seed: int) -> tuple[dict[str, str], dict[str, Any]]:
    qid_group: dict[str, str] = {}
    for language, rows in rows_by_language.items():
        qid_group.update(connected_query_groups(rows, language))
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid, group in qid_group.items():
        grouped[group].append(qid)
    rng = random.Random(seed)
    buckets = list(grouped.items())
    rng.shuffle(buckets)
    buckets.sort(key=lambda row: len(row[1]), reverse=True)
    targets = {"train": 0.70, "dev": 0.15, "test": 0.15}
    total = sum(len(qids) for _, qids in buckets)
    counts = Counter()
    group_split: dict[str, str] = {}
    for group, qids in buckets:
        split = min(targets, key=lambda name: counts[name] / max(1.0, targets[name] * total))
        group_split[group] = split
        counts[split] += len(qids)
    qid_split = {qid: group_split[group] for qid, group in qid_group.items()}
    summary = {
        "seed": seed,
        "strategy": "connected gold-report group-aware 70/15/15",
        "query_count": len(qid_split),
        "group_count": len(grouped),
        "split_query_counts": dict(counts),
        "split_group_counts": dict(Counter(group_split.values())),
        "group_overlap": {
            f"{a}_{b}": len({g for g, s in group_split.items() if s == a} & {g for g, s in group_split.items() if s == b})
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
        },
        "queries": {qid: {"group_id": qid_group[qid], "split": split} for qid, split in qid_split.items()},
    }
    return qid_split, summary


def oracle_recall(rows: list[dict[str, Any]], run: str, language: str, max_available: int, query_type: str = "all") -> list[dict[str, Any]]:
    out = []
    for depth in DEPTHS:
        if depth > max_available:
            out.append({"language": language, "query_type": query_type, "candidate_source": run, "depth": depth, "oracle_recall": "NA", "query_count": len(rows), "note": f"only top-{max_available} cached"})
            continue
        recalls = []
        for row in rows:
            gold = {cid for cid, rel in row.get("qrels", {}).items() if int(rel) > 0}
            found = {cand["corpus_id"] for cand in row.get("candidates", [])[:depth]}
            recalls.append(len(gold & found) / max(1, len(gold)))
        out.append({"language": language, "query_type": query_type, "candidate_source": run, "depth": depth, "oracle_recall": sum(recalls) / len(recalls), "query_count": len(rows), "note": "cached candidates"})
    return out


def conflict_labels(query: str, constraints: dict[str, Any], wrong_text: str, wrong_cid: str, gold_cids: set[str], col_candidates: list[dict[str, Any]]) -> list[str]:
    labels = []
    years = constraints["year_terms"]
    page_years = set(YEAR_RE.findall(wrong_text))
    if years and not set(years).issubset(page_years) and page_years:
        labels.append("year_conflict")
    metric_terms = []
    for name in constraints["canonical_metrics"]:
        metric_terms.extend(FIN_METRICS[name]["aliases"])
    if metric_terms and not find_terms(wrong_text, metric_terms):
        labels.append("metric_conflict")
    entity = constraints.get("entity")
    if entity and constraints.get("entity_confidence", 0) >= 0.8:
        tokens = [token.lower() for token in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", entity)]
        if tokens and not any(token in wrong_text for token in tokens):
            labels.append("entity_conflict")
    if constraints["period_terms"] and not find_terms(wrong_text, constraints["period_terms"]):
        labels.append("period_conflict")
    if constraints["unit_terms"] and not find_terms(wrong_text, constraints["unit_terms"]):
        labels.append("unit_conflict")
    statement = constraints["statement_type"]
    if statement != "unknown" and not find_terms(wrong_text, STATEMENTS.get(statement, [])):
        labels.append("statement_conflict")
    gold_scores = [float(c["score"]) for c in col_candidates if c["corpus_id"] in gold_cids]
    wrong_score = next((float(c["score"]) for c in col_candidates if c["corpus_id"] == wrong_cid), None)
    if gold_scores and wrong_score is not None and wrong_score >= 0.9 * max(gold_scores):
        labels.append("visual_similarity_interference")
    return labels or ["unresolved"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/retrieval/rerun_tab_safe_20260624_074152"))
    parser.add_argument("--metric-dir", type=Path, default=Path("artifacts/retrieval/financial_metric_aware_reranking_20260630_074825"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    rows_by_language: dict[str, list[dict[str, Any]]] = {}
    mono_by_language: dict[str, list[dict[str, Any]]] = {}
    all_wanted: dict[str, set[str]] = {}
    oracle_rows: list[dict[str, Any]] = []
    for language in ("en", "zh"):
        col = read_jsonl(args.baseline_dir / f"colqwen2_{language}_predictions.jsonl")
        mono = read_jsonl(args.baseline_dir / f"colqwen2_{language}_monoqwen2vl_predictions_raw.jsonl")
        rows_by_language[language] = col
        mono_by_language[language] = mono
        oracle_rows += oracle_recall(col, "ColQwen2", language, 100)
        oracle_rows += oracle_recall(mono, "MonoQwen2-VL", language, 20)
        all_wanted[language] = {c["corpus_id"] for row in col for c in row["candidates"][:100]}
        all_wanted[language].update(cid for row in col for cid in row.get("qrels", {}))

    qid_split, split_manifest = assign_group_split(rows_by_language, args.seed)
    (args.run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    constraint_rows = []
    sample_lines = ["# FinConRank Query Constraint Samples", "", "The labels below are automatically parsed, not manually annotated.", ""]
    taxonomy_rows = []
    for language in ("en", "zh"):
        page_text = select_page_texts(args.index_dir / f"{language}_pages.jsonl", all_wanted[language])
        col_by_qid = {row["query_id"]: row for row in rows_by_language[language]}
        mono_by_qid = {row["query_id"]: row for row in mono_by_language[language]}
        for index, row in enumerate(rows_by_language[language]):
            constraints = extract_constraints(row.get("query", ""))
            qtype = classify_query(row.get("query", ""), constraints)
            item = {"query_id": row["query_id"], "language": language, "split": qid_split[row["query_id"]], "group_id": split_manifest["queries"][row["query_id"]]["group_id"], "query": row.get("query", ""), "query_type": qtype, "constraints": constraints}
            constraint_rows.append(item)
            if index < 50:
                sample_lines.extend([f"## {language.upper()} {index + 1}: `{row['query_id']}`", "", row.get("query", ""), "", "```json", json.dumps(constraints, ensure_ascii=False, indent=2), "```", ""])
            mono_row = mono_by_qid[row["query_id"]]
            gold = {cid for cid, rel in row.get("qrels", {}).items() if int(rel) > 0}
            mono_candidates = mono_row.get("candidates", [])
            top = mono_candidates[0] if mono_candidates else {}
            if top.get("corpus_id") and top["corpus_id"] not in gold:
                labels = conflict_labels(row.get("query", ""), constraints, page_text.get(top["corpus_id"], ""), top["corpus_id"], gold, col_by_qid[row["query_id"]]["candidates"])
                taxonomy_rows.append({"language": language, "query_id": row["query_id"], "query_type": qtype, "split": qid_split[row["query_id"]], "wrong_top1": top["corpus_id"], "gold_pages": "|".join(sorted(gold)), "conflict_types": "|".join(labels), "mono_gold_rank": first_gold_rank(mono_row["candidates"], row.get("qrels", {})) or 999})
    write_jsonl(args.run_dir / "query_constraints.jsonl", constraint_rows)
    (args.run_dir / "query_constraint_samples.md").write_text("\n".join(sample_lines), encoding="utf-8")
    write_csv(args.run_dir / "baseline_error_taxonomy.csv", taxonomy_rows)
    constraints_by_qid = {row["query_id"]: row for row in constraint_rows}
    for language in ("en", "zh"):
        col_by_qid = {row["query_id"]: row for row in rows_by_language[language]}
        mono_by_qid = {row["query_id"]: row for row in mono_by_language[language]}
        types = sorted({row["query_type"] for row in constraint_rows if row["language"] == language})
        for query_type in types:
            qids = {qid for qid, item in constraints_by_qid.items() if item["language"] == language and item["query_type"] == query_type}
            oracle_rows += oracle_recall([col_by_qid[qid] for qid in qids], "ColQwen2", language, 100, query_type)
            oracle_rows += oracle_recall([mono_by_qid[qid] for qid in qids], "MonoQwen2-VL", language, 20, query_type)
    write_csv(args.run_dir / "oracle_recall_by_depth.csv", oracle_rows)

    summary = {
        "run_dir": str(args.run_dir),
        "baseline_dir": str(args.baseline_dir),
        "metric_dir": str(args.metric_dir),
        "index_dir": str(args.index_dir),
        "query_counts": {language: len(rows) for language, rows in rows_by_language.items()},
        "candidate_depths": {"colqwen2": 100, "monoqwen2vl": 20},
        "constraint_rows": len(constraint_rows),
        "baseline_top1_errors": len(taxonomy_rows),
        "conflict_counts": dict(Counter(label for row in taxonomy_rows for label in row["conflict_types"].split("|"))),
        "data_path_declared": release_path('FINRAGBENCH_DATA_ROOT', 'data/FinRAGBench-V', ''),
        "data_path_declared_exists": Path(release_path('FINRAGBENCH_DATA_ROOT', 'data/FinRAGBench-V', '')).exists(),
        "actual_read_only_source": release_path('FINRAGBENCH_DATA_ROOT', 'data/FinRAGBench-V', ''),
        "prepared_used": False,
    }
    (args.run_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
