#!/usr/bin/env python3
"""Financial Metric-Aware Evidence Reranking for FinRAGBench-V retrieval.

The module reranks the existing ColQwen2 + MonoQwen2-VL strong candidates with
financial-domain features extracted from the query and OCR/PDF page text.
Learned reranking uses query-level outer-fold OOF evaluation. Relevance labels
are used only for train/dev weight selection and final metric computation, not
as scoring features.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10"]
QTYPE_ORDER = [
    "numeric_lookup",
    "table_comparison",
    "ratio_growth",
    "cash_flow",
    "balance_sheet",
    "income_statement",
    "other_financial",
]


FINANCIAL_METRICS: dict[str, dict[str, Any]] = {
    "revenue": {
        "statement": "income_statement",
        "aliases": ["revenue", "revenues", "sales", "turnover", "net sales", "营业收入", "收入", "营业总收入"],
    },
    "operating_income": {
        "statement": "income_statement",
        "aliases": ["operating income", "operating profit", "income from operations", "营业利润", "经营利润"],
    },
    "net_income": {
        "statement": "income_statement",
        "aliases": ["net income", "net profit", "net earnings", "profit attributable", "归母净利润", "净利润", "归属于母公司股东的净利润"],
    },
    "gross_profit": {
        "statement": "income_statement",
        "aliases": ["gross profit", "gross income", "毛利", "毛利润"],
    },
    "eps": {
        "statement": "income_statement",
        "aliases": ["eps", "earnings per share", "diluted eps", "basic eps", "每股收益", "基本每股收益", "稀释每股收益"],
    },
    "total_assets": {
        "statement": "balance_sheet",
        "aliases": ["total assets", "assets", "总资产", "资产总计"],
    },
    "liabilities": {
        "statement": "balance_sheet",
        "aliases": ["liabilities", "total liabilities", "负债", "负债合计", "负债总计"],
    },
    "equity": {
        "statement": "balance_sheet",
        "aliases": ["equity", "shareholders' equity", "stockholders' equity", "owners' equity", "股东权益", "所有者权益", "权益总计"],
    },
    "cash": {
        "statement": "balance_sheet",
        "aliases": ["cash", "cash and equivalents", "cash equivalents", "货币资金", "现金及现金等价物", "现金"],
    },
    "inventory": {
        "statement": "balance_sheet",
        "aliases": ["inventory", "inventories", "存货"],
    },
    "receivables": {
        "statement": "balance_sheet",
        "aliases": ["receivables", "accounts receivable", "trade receivables", "应收账款", "应收款项"],
    },
    "operating_cash_flow": {
        "statement": "cash_flow",
        "aliases": ["operating cash flow", "cash flows from operating activities", "net cash provided by operating activities", "经营活动现金流", "经营活动产生的现金流量净额"],
    },
    "investing_cash_flow": {
        "statement": "cash_flow",
        "aliases": ["investing cash flow", "cash flows from investing activities", "投资活动现金流", "投资活动产生的现金流量净额"],
    },
    "financing_cash_flow": {
        "statement": "cash_flow",
        "aliases": ["financing cash flow", "cash flows from financing activities", "筹资活动现金流", "筹资活动产生的现金流量净额"],
    },
    "free_cash_flow": {
        "statement": "cash_flow",
        "aliases": ["free cash flow", "fcf", "自由现金流"],
    },
    "gross_margin": {
        "statement": "income_statement",
        "aliases": ["gross margin", "gross profit margin", "毛利率"],
    },
    "net_margin": {
        "statement": "income_statement",
        "aliases": ["net margin", "net profit margin", "净利率", "净利润率"],
    },
    "roe": {
        "statement": "balance_sheet",
        "aliases": ["roe", "return on equity", "净资产收益率"],
    },
    "roa": {
        "statement": "balance_sheet",
        "aliases": ["roa", "return on assets", "资产收益率", "总资产收益率"],
    },
    "debt_ratio": {
        "statement": "balance_sheet",
        "aliases": ["debt ratio", "debt-to-assets", "liabilities to assets", "资产负债率"],
    },
}

STATEMENT_TERMS = {
    "income_statement": [
        "income statement",
        "statement of operations",
        "statement of earnings",
        "consolidated statements of income",
        "利润表",
        "损益表",
        "综合收益表",
    ],
    "balance_sheet": [
        "balance sheet",
        "statement of financial position",
        "consolidated balance sheets",
        "资产负债表",
    ],
    "cash_flow": [
        "cash flow",
        "statements of cash flows",
        "statement of cash flows",
        "现金流量表",
        "现金流",
    ],
    "notes": ["notes to", "note ", "附注", "注释"],
}

UNIT_TERMS = [
    "million",
    "millions",
    "billion",
    "billions",
    "thousand",
    "thousands",
    "usd",
    "rmb",
    "$",
    "%",
    "percent",
    "percentage",
    "百万元",
    "千元",
    "万元",
    "亿元",
    "人民币",
    "美元",
]

OPERATION_TERMS = [
    "increase",
    "decrease",
    "compare",
    "comparison",
    "difference",
    "growth",
    "ratio",
    "trend",
    "change",
    "higher",
    "lower",
    "占比",
    "比较",
    "增长",
    "下降",
    "差异",
    "趋势",
    "比例",
    "变化",
    "增加",
    "减少",
]

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
NUM_RE = re.compile(r"[$¥]?\(?-?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|percent|million|billion|thousand|百万元|千元|万元|亿元)?", re.I)
TOKEN_RE = re.compile(r"[a-z0-9%$¥]+|[\u4e00-\u9fff]+", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_by_qid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in rows}


def normalize_values(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def evaluate(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> dict[str, float]:
    rels = [1 if cand["corpus_id"] in qrels and qrels[cand["corpus_id"]] > 0 else 0 for cand in candidates]
    total_rel = max(1, sum(1 for value in qrels.values() if value > 0))

    def dcg_at(k: int) -> float:
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels[:k]))

    def ideal_dcg_at(k: int) -> float:
        return sum(1.0 / math.log2(idx + 2) for idx in range(min(total_rel, k)))

    out = {}
    for k in (5, 10):
        ideal = ideal_dcg_at(k)
        out[f"nDCG@{k}"] = dcg_at(k) / ideal if ideal else 0.0
        out[f"Recall@{k}"] = sum(rels[:k]) / total_rel
    mrr = 0.0
    for idx, rel in enumerate(rels[:10], start=1):
        if rel:
            mrr = 1.0 / idx
            break
    out["MRR@10"] = mrr
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: sum(row["metrics"][metric] for row in rows) / len(rows) for metric in METRICS}


def first_gold_rank(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> int | None:
    for idx, cand in enumerate(candidates, start=1):
        if cand["corpus_id"] in qrels and qrels[cand["corpus_id"]] > 0:
            return idx
    return None


def contains_term(text: str, term: str) -> bool:
    term_l = term.lower()
    if re.search(r"[a-z0-9]", term_l):
        return re.search(rf"(?<![a-z0-9]){re.escape(term_l)}(?![a-z0-9])", text) is not None
    return term_l in text


def find_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if contains_term(text, term)]


def classify_financial_query(query: str, constraints: dict[str, Any]) -> str:
    text = query.lower()
    stmt = constraints["statement_type"]
    if any(term in text for term in ["ratio", "margin", "roe", "roa", "growth", "%", "比例", "率", "增长", "占比"]):
        return "ratio_growth"
    if stmt == "cash_flow":
        return "cash_flow"
    if stmt == "balance_sheet":
        return "balance_sheet"
    if stmt == "income_statement":
        return "income_statement"
    if any(term in text for term in ["compare", "between", "higher", "lower", "difference", "比较", "高于", "低于", "差异"]):
        return "table_comparison"
    if constraints["metric_terms"] or constraints["year_terms"] or NUM_RE.search(text):
        return "numeric_lookup"
    return "other_financial"


def extract_query_constraints(query: str) -> dict[str, Any]:
    query = "" if query is None else str(query)
    text = query.lower()
    metric_terms = []
    canonical_metrics = []
    statement_votes = Counter()
    for canonical, spec in FINANCIAL_METRICS.items():
        matched = find_terms(text, spec["aliases"])
        if matched:
            canonical_metrics.append(canonical)
            metric_terms.extend(matched)
            statement_votes[spec["statement"]] += 1

    explicit_statement = "unknown"
    for statement, terms in STATEMENT_TERMS.items():
        if find_terms(text, terms):
            explicit_statement = statement
            statement_votes[statement] += 2
            break
    if statement_votes:
        statement_type = statement_votes.most_common(1)[0][0]
    else:
        statement_type = explicit_statement

    constraints = {
        "canonical_metrics": canonical_metrics,
        "metric_terms": sorted(set(metric_terms), key=len, reverse=True),
        "year_terms": sorted(set(YEAR_RE.findall(query))),
        "unit_terms": sorted(set(find_terms(text, UNIT_TERMS))),
        "operation_terms": sorted(set(find_terms(text, OPERATION_TERMS))),
        "statement_type": statement_type,
    }
    constraints["query_type"] = classify_financial_query(query, constraints)
    return constraints


def load_page_texts(index_dir: Path, prefix: str) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    page_text: dict[str, str] = {}
    page_title: dict[str, str] = {}
    block_text: dict[str, list[str]] = defaultdict(list)
    with (index_dir / f"{prefix}_pages.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cid = row["corpus_id"]
            page_title[cid] = str(row.get("title", "") or "")
            page_text[cid] = f"{row.get('title', '')} {row.get('text', '')}".strip()
    with (index_dir / f"{prefix}_blocks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = str(row.get("text", "") or "")
            if text:
                block_text[row["corpus_id"]].append(text)
    return page_text, page_title, block_text


def nearby_score(text: str, anchors: list[str], targets: list[str], window: int = 140) -> float:
    if not anchors or not targets:
        return 0.0
    positions = []
    for anchor in anchors:
        start = text.find(anchor.lower())
        while start >= 0:
            positions.append(start)
            start = text.find(anchor.lower(), start + 1)
    if not positions:
        return 0.0
    best = 0.0
    for pos in positions[:20]:
        span = text[max(0, pos - window) : pos + window]
        hits = sum(1 for target in targets if contains_term(span, target))
        best = max(best, min(1.0, hits / max(1, len(targets))))
    return best


def numeric_context_score(text: str, metric_terms: list[str]) -> float:
    if not metric_terms:
        return 0.0
    best = 0.0
    for term in metric_terms:
        start = text.find(term.lower())
        while start >= 0:
            span = text[max(0, start - 100) : start + 180]
            if NUM_RE.search(span):
                best = 1.0
                break
            start = text.find(term.lower(), start + 1)
        if best:
            break
    return best


def table_density_score(text: str) -> float:
    token_count = max(1, len(TOKEN_RE.findall(text)))
    num_count = len(NUM_RE.findall(text))
    year_count = len(YEAR_RE.findall(text))
    percent_count = text.count("%") + text.count("percent") + text.count("百分")
    line_count = max(1, text.count("\n") + 1)
    numeric_density = min(1.0, num_count / max(8.0, token_count * 0.12))
    year_density = min(1.0, year_count / 4.0)
    pct_density = min(1.0, percent_count / 4.0)
    line_bonus = min(1.0, line_count / 25.0)
    return 0.55 * numeric_density + 0.20 * year_density + 0.15 * pct_density + 0.10 * line_bonus


def page_financial_features(text: str, title: str, blocks: list[str], constraints: dict[str, Any]) -> dict[str, float]:
    combined = f"{title} {text} {' '.join(blocks[:80])}".lower()
    query_metric_terms = constraints["metric_terms"]
    if not query_metric_terms and constraints["canonical_metrics"]:
        for metric in constraints["canonical_metrics"]:
            query_metric_terms.extend(FINANCIAL_METRICS[metric]["aliases"])

    metric_hits = find_terms(combined, query_metric_terms)
    metric_match = min(1.0, len(set(metric_hits)) / max(1, min(3, len(set(query_metric_terms)))))

    years = constraints["year_terms"]
    adjacent_years = []
    for year in years:
        y = int(year)
        adjacent_years.extend([str(y - 1), str(y + 1)])
    exact_year_hits = find_terms(combined, years)
    adjacent_hits = find_terms(combined, adjacent_years)
    year_match = 0.0
    if years:
        year_match = min(1.0, len(set(exact_year_hits)) / len(set(years)))
        if year_match < 1.0 and adjacent_hits:
            year_match = max(year_match, 0.4)

    statement = constraints["statement_type"]
    statement_match = 0.0
    if statement != "unknown":
        statement_match = 1.0 if find_terms(combined, STATEMENT_TERMS.get(statement, [])) else 0.0

    unit_terms = constraints["unit_terms"]
    unit_match = min(1.0, len(set(find_terms(combined, unit_terms))) / max(1, len(set(unit_terms)))) if unit_terms else 0.0
    table_density = table_density_score(combined)
    numeric_context = numeric_context_score(combined, query_metric_terms)
    co_targets = list(years) + list(unit_terms)
    if not co_targets:
        co_targets = [m.group(0) for m in NUM_RE.finditer(combined[:1000])][:2]
    cooccurrence = nearby_score(combined, query_metric_terms, co_targets)
    return {
        "metric_match_score": metric_match,
        "year_match_score": year_match,
        "statement_match_score": statement_match,
        "unit_match_score": unit_match,
        "table_density_score": table_density,
        "cooccurrence_score": cooccurrence,
        "numeric_context_score": numeric_context,
    }


def build_items(
    strong_row: dict[str, Any],
    evidence_row: dict[str, Any],
    page_text: dict[str, str],
    page_title: dict[str, str],
    block_text: dict[str, list[str]],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    strong_candidates = strong_row["candidates"]
    strong_ids = [cand["corpus_id"] for cand in strong_candidates]
    strong_rank = {cid: idx for idx, cid in enumerate(strong_ids, start=1)}
    visual_norm = normalize_values({cand["corpus_id"]: float(cand.get("score", 0.0)) for cand in strong_candidates})
    mono_norm = normalize_values({cand["corpus_id"]: float(cand.get("rerank_score", cand.get("score", 0.0))) for cand in strong_candidates})
    ev_candidates = evidence_row.get("candidates", []) if evidence_row else []
    ev_source = {cand["corpus_id"]: cand.get("source_scores", {}) for cand in ev_candidates}
    ev_blocks = {cand["corpus_id"]: [b.get("text", "") for b in cand.get("block_contributions", [])] for cand in ev_candidates}
    items = []
    for cid in strong_ids:
        rank = strong_rank[cid]
        blocks = list(block_text.get(cid, []))
        blocks.extend(ev_blocks.get(cid, []))
        fin = page_financial_features(page_text.get(cid, ""), page_title.get(cid, ""), blocks, constraints)
        source = ev_source.get(cid, {})
        rank_prior = 1.0 / math.log2(rank + 1.0)
        items.append(
            {
                "corpus_id": cid,
                "strong_rank": rank,
                "rank_prior": rank_prior,
                "visual": visual_norm.get(cid, 0.0),
                "mono": mono_norm.get(cid, 0.0),
                "text": float(source.get("text", 0.0) or 0.0),
                "block": float(source.get("block", 0.0) or 0.0),
                "query_type": constraints["query_type"],
                **fin,
            }
        )
    return items


def feature_vector(item: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            item["rank_prior"],
            item["mono"],
            item["visual"],
            item["text"],
            item["block"],
            item["metric_match_score"],
            item["year_match_score"],
            item["statement_match_score"],
            item["unit_match_score"],
            item["table_density_score"],
            item["cooccurrence_score"],
            item["numeric_context_score"],
        ],
        dtype=float,
    )


def score_rule(item: dict[str, Any]) -> float:
    return (
        2.0 * item["rank_prior"]
        + 0.35 * item["mono"]
        + 0.12 * item["visual"]
        + 0.10 * item["metric_match_score"]
        + 0.07 * item["year_match_score"]
        + 0.06 * item["statement_match_score"]
        + 0.035 * item["unit_match_score"]
        + 0.045 * item["cooccurrence_score"]
        + 0.045 * item["numeric_context_score"]
        + 0.025 * item["table_density_score"]
    )


def score_gate(item: dict[str, Any], query_type: str, max_conf: float) -> float:
    base = 2.0 * item["rank_prior"] + 0.35 * item["mono"] + 0.12 * item["visual"]
    dimensions = sum(
        [
            item["metric_match_score"] >= 0.5,
            item["year_match_score"] >= 0.5,
            item["statement_match_score"] >= 0.5,
            item["cooccurrence_score"] >= 0.4,
            item["numeric_context_score"] >= 0.5,
        ]
    )
    if max_conf < 1.35 or dimensions < 2:
        return base
    type_boost = 0.04 if query_type in {"numeric_lookup", "ratio_growth", "table_comparison"} else 0.0
    return base + type_boost + 0.09 * item["metric_match_score"] + 0.08 * item["year_match_score"] + 0.06 * item["statement_match_score"] + 0.04 * item["cooccurrence_score"]


def rank_items(items: list[dict[str, Any]], scores: list[float], run_name: str, constraints: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = sorted(zip(items, scores), key=lambda pair: (-pair[1], pair[0]["strong_rank"], pair[0]["corpus_id"]))
    rows = []
    for rank, (item, score) in enumerate(ranked[:100], start=1):
        rows.append(
            {
                "corpus_id": item["corpus_id"],
                "rank": rank,
                "score": float(score),
                "run_name": run_name,
                "financial_features": {
                    key: item[key]
                    for key in [
                        "metric_match_score",
                        "year_match_score",
                        "statement_match_score",
                        "unit_match_score",
                        "table_density_score",
                        "cooccurrence_score",
                        "numeric_context_score",
                    ]
                },
                "base_features": {
                    "strong_rank": item["strong_rank"],
                    "rank_prior": item["rank_prior"],
                    "mono": item["mono"],
                    "visual": item["visual"],
                    "text": item["text"],
                    "block": item["block"],
                },
                "query_constraints": constraints,
            }
        )
    return rows


def run_fixed_variant(
    query_items: dict[str, list[dict[str, Any]]],
    strong_by_qid: dict[str, dict[str, Any]],
    constraints_by_qid: dict[str, dict[str, Any]],
    variant: str,
    language: str,
) -> list[dict[str, Any]]:
    rows = []
    for qid, items in query_items.items():
        constraints = constraints_by_qid[qid]
        if variant == "metric_rule_boost":
            scores = [score_rule(item) for item in items]
        elif variant == "metric_confidence_gate":
            max_conf = max(
                (
                    item["metric_match_score"]
                    + item["year_match_score"]
                    + item["statement_match_score"]
                    + item["cooccurrence_score"]
                    + item["numeric_context_score"]
                    for item in items
                ),
                default=0.0,
            )
            scores = [score_gate(item, constraints["query_type"], max_conf) for item in items]
        else:
            raise ValueError(variant)
        src = strong_by_qid[qid]
        candidates = rank_items(items, scores, f"financial_metric_aware_{variant}_{language}", constraints)
        rows.append(
            {
                "query_id": qid,
                "query": src.get("query"),
                "answer": src.get("answer"),
                "qrels": src.get("qrels", {}),
                "query_type": constraints["query_type"],
                "query_constraints": constraints,
                "metrics": evaluate(candidates, src.get("qrels", {})),
                "candidates": candidates,
            }
        )
    return rows


def split_outer_folds(qids: list[str], seed: int, folds: int) -> list[list[str]]:
    shuffled = qids[:]
    random.Random(seed).shuffle(shuffled)
    return [shuffled[idx::folds] for idx in range(folds)]


def candidate_weights(dim: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    base = [2.0, 0.35, 0.12, 0.04, 0.04, 0.08, 0.06, 0.05, 0.03, 0.02, 0.04, 0.05]
    weights = [np.asarray(base[:dim], dtype=float)]
    for _ in range(220):
        row = [
            rng.uniform(1.3, 2.4),
            rng.uniform(0.15, 0.65),
            rng.uniform(0.05, 0.25),
            rng.uniform(0.0, 0.10),
            rng.uniform(0.0, 0.12),
            rng.uniform(0.0, 0.22),
            rng.uniform(0.0, 0.18),
            rng.uniform(0.0, 0.16),
            rng.uniform(0.0, 0.08),
            rng.uniform(0.0, 0.08),
            rng.uniform(0.0, 0.14),
            rng.uniform(0.0, 0.16),
        ]
        weights.append(np.asarray(row[:dim], dtype=float))
    return weights


def mean_metric(qids: list[str], query_items: dict[str, list[dict[str, Any]]], strong_by_qid: dict[str, dict[str, Any]], constraints_by_qid: dict[str, dict[str, Any]], weights: np.ndarray) -> float:
    vals = []
    for qid in qids:
        items = query_items[qid]
        scores = [float(weights @ feature_vector(item)) for item in items]
        candidates = rank_items(items, scores, "financial_metric_aware_learned_tmp", constraints_by_qid[qid])
        vals.append(evaluate(candidates, strong_by_qid[qid].get("qrels", {}))["nDCG@5"])
    return sum(vals) / len(vals) if vals else 0.0


def run_learned_oof(
    query_items: dict[str, list[dict[str, Any]]],
    strong_by_qid: dict[str, dict[str, Any]],
    constraints_by_qid: dict[str, dict[str, Any]],
    language: str,
    seed: int,
    folds: int,
    dev_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qids = list(query_items)
    outer = split_outer_folds(qids, seed, folds)
    first_items = next(items for items in query_items.values() if items)
    all_weights = candidate_weights(len(feature_vector(first_items[0])), seed + (0 if language == "en" else 1000))
    predictions = []
    fold_summary = []
    for fold_id, test_qids in enumerate(outer):
        remaining = [qid for qid in qids if qid not in set(test_qids)]
        random.Random(seed + fold_id + 17).shuffle(remaining)
        n_dev = max(1, int(0.2 * len(remaining)))
        dev_qids = remaining[:n_dev]
        train_qids = remaining[n_dev:]
        train_scores = [(mean_metric(train_qids, query_items, strong_by_qid, constraints_by_qid, w), idx, w) for idx, w in enumerate(all_weights)]
        train_scores.sort(reverse=True, key=lambda row: row[0])
        shortlist = train_scores[:dev_candidates]
        dev_scores = [(mean_metric(dev_qids, query_items, strong_by_qid, constraints_by_qid, w), train_score, idx, w) for train_score, idx, w in shortlist]
        dev_scores.sort(reverse=True, key=lambda row: (row[0], row[1]))
        dev_score, train_score, weight_idx, best_w = dev_scores[0]
        fold_summary.append(
            {
                "fold": fold_id,
                "train_queries": len(train_qids),
                "dev_queries": len(dev_qids),
                "test_queries": len(test_qids),
                "selected_weight_index": weight_idx,
                "selected_train_nDCG@5": train_score,
                "selected_dev_nDCG@5": dev_score,
                "selected_weights": best_w.tolist(),
                "test_qids": test_qids,
            }
        )
        for qid in test_qids:
            items = query_items[qid]
            constraints = constraints_by_qid[qid]
            scores = [float(best_w @ feature_vector(item)) for item in items]
            src = strong_by_qid[qid]
            candidates = rank_items(items, scores, f"financial_metric_aware_metric_learned_rerank_{language}", constraints)
            predictions.append(
                {
                    "query_id": qid,
                    "query": src.get("query"),
                    "answer": src.get("answer"),
                    "qrels": src.get("qrels", {}),
                    "query_type": constraints["query_type"],
                    "query_constraints": constraints,
                    "metrics": evaluate(candidates, src.get("qrels", {})),
                    "candidates": candidates,
                }
            )
    predictions.sort(key=lambda row: qids.index(row["query_id"]))
    summary = {
        "seed": seed,
        "outer_folds": folds,
        "candidate_weight_count": len(all_weights),
        "dev_candidates": dev_candidates,
        "folds": fold_summary,
    }
    return predictions, summary


def subset_metrics(rows: list[dict[str, Any]], run_name: str, language: str) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("query_type", "other_financial")].append(row)
    out = []
    for qtype in QTYPE_ORDER:
        selected = by_type.get(qtype, [])
        if not selected:
            continue
        out.append({"language": language, "run": run_name, "query_type": qtype, "query_count": len(selected), **aggregate(selected)})
    return out


def audit_rows(query_items: dict[str, list[dict[str, Any]]], constraints_by_qid: dict[str, dict[str, Any]], language: str) -> list[dict[str, Any]]:
    rows = []
    def avg(top: list[dict[str, Any]], key: str) -> float:
        return sum(x[key] for x in top) / len(top) if top else 0.0

    for qid, items in query_items.items():
        constraints = constraints_by_qid[qid]
        top = items[:10]
        rows.append(
            {
                "language": language,
                "query_id": qid,
                "query_type": constraints["query_type"],
                "canonical_metrics": "|".join(constraints["canonical_metrics"]),
                "metric_terms": "|".join(constraints["metric_terms"]),
                "year_terms": "|".join(constraints["year_terms"]),
                "unit_terms": "|".join(constraints["unit_terms"]),
                "statement_type": constraints["statement_type"],
                "top10_avg_metric_match": avg(top, "metric_match_score"),
                "top10_avg_year_match": avg(top, "year_match_score"),
                "top10_avg_statement_match": avg(top, "statement_match_score"),
                "top10_avg_table_density": avg(top, "table_density_score"),
                "top10_avg_cooccurrence": avg(top, "cooccurrence_score"),
                "top10_avg_numeric_context": avg(top, "numeric_context_score"),
            }
        )
    return rows


def build_language_inputs(args: argparse.Namespace, language: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = "en" if language == "en" else "zh"
    strong_rows = read_jsonl(args.strong_base_dir / f"colqwen2_{prefix}_monoqwen2vl_predictions_raw.jsonl")
    evidence_rows = read_jsonl(args.ocr_rerun_dir / f"{prefix}_query_router_multi_granularity_mono_predictions.jsonl")
    page_text, page_title, block_text = load_page_texts(args.index_dir, prefix)
    evidence_by_qid = row_by_qid(evidence_rows)
    strong_by_qid = row_by_qid(strong_rows)
    constraints_by_qid = {row["query_id"]: extract_query_constraints(row.get("query", "")) for row in strong_rows}
    query_items = {
        row["query_id"]: build_items(row, evidence_by_qid.get(row["query_id"], {}), page_text, page_title, block_text, constraints_by_qid[row["query_id"]])
        for row in strong_rows
    }
    reported_strong = json.loads((args.strong_base_dir / f"colqwen2_{prefix}_monoqwen2vl_metrics.json").read_text(encoding="utf-8"))
    return strong_by_qid, query_items, constraints_by_qid, reported_strong


def strong_rows_from_inputs(strong_by_qid: dict[str, dict[str, Any]], constraints_by_qid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for qid, row in strong_by_qid.items():
        candidates = row["candidates"]
        rows.append(
            {
                "query_id": qid,
                "query": row.get("query"),
                "answer": row.get("answer"),
                "qrels": row.get("qrels", {}),
                "query_type": constraints_by_qid[qid]["query_type"],
                "query_constraints": constraints_by_qid[qid],
                "metrics": evaluate(candidates, row.get("qrels", {})),
                "candidates": candidates,
            }
        )
    return rows


def write_case_studies(path: Path, strong_rows: list[dict[str, Any]], variant_rows: dict[str, list[dict[str, Any]]]) -> None:
    strong_by = row_by_qid(strong_rows)
    lines = ["# Financial Metric-Aware Reranking Case Studies", ""]
    for name, rows in variant_rows.items():
        diffs = []
        for row in rows:
            base = strong_by[row["query_id"]]
            br = first_gold_rank(base["candidates"], base["qrels"]) or 999
            nr = first_gold_rank(row["candidates"], row["qrels"]) or 999
            diffs.append((br - nr, row, br, nr))
        for title, selected in [("Improvements", sorted(diffs, reverse=True, key=lambda x: x[0])[:5]), ("Regressions", sorted(diffs, key=lambda x: x[0])[:5])]:
            lines.extend([f"## {name} {title}", ""])
            for delta, row, br, nr in selected:
                lines.append(f"- query_id: `{row['query_id']}`")
                lines.append(f"  query_type: `{row.get('query_type')}`")
                lines.append(f"  query: {row.get('query')}")
                lines.append(f"  gold rank strong -> variant: {br} -> {nr} (delta {delta})")
                lines.append(f"  constraints: `{json.dumps(row.get('query_constraints', {}), ensure_ascii=False)}`")
                lines.append(f"  strong top5: `{[c['corpus_id'] for c in strong_by[row['query_id']]['candidates'][:5]]}`")
                lines.append(f"  variant top5: `{[c['corpus_id'] for c in row['candidates'][:5]]}`")
                lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-base-dir", type=Path, default=Path("artifacts/retrieval/rerun_tab_safe_20260624_074152"))
    parser.add_argument("--ocr-rerun-dir", type=Path, default=Path("artifacts/retrieval/real_router_multigranularity_20260624_131957/candidate_ocr_pdf_merged_rerun"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged"))
    parser.add_argument("--query-aware-confirmatory-dir", type=Path, default=Path("artifacts/retrieval/query_aware_calibration_confirmatory_20260630_072446"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dev-candidates", type=int, default=25)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    delta_rows = []
    subset_rows = []
    audit_all = []
    all_case_variants: dict[str, list[dict[str, Any]]] = {}
    split_summary = {"seed": args.seed, "outer_folds": args.folds, "languages": {}}

    for language in ("en", "zh"):
        strong_by_qid, query_items, constraints_by_qid, reported_strong = build_language_inputs(args, language)
        strong_eval_rows = strong_rows_from_inputs(strong_by_qid, constraints_by_qid)
        strong_metrics = aggregate(strong_eval_rows)
        metric_rows.append({"language": language, "run": "reported_strong_full_dataset", **reported_strong})
        metric_rows.append({"language": language, "run": "recomputed_strong_full_dataset", **strong_metrics})
        subset_rows.extend(subset_metrics(strong_eval_rows, "recomputed_strong_full_dataset", language))

        variants = {}
        for variant in ("metric_rule_boost", "metric_confidence_gate"):
            rows = run_fixed_variant(query_items, strong_by_qid, constraints_by_qid, variant, language)
            variants[variant] = rows
            write_jsonl(args.out_dir / f"{language}_{variant}_predictions.jsonl", rows)
            metrics = aggregate(rows)
            metric_rows.append({"language": language, "run": variant, **metrics})
            delta_rows.append({"language": language, "run": variant, "baseline": "recomputed_strong_full_dataset", **{f"delta_{m}": metrics[m] - strong_metrics[m] for m in METRICS}})
            subset_rows.extend(subset_metrics(rows, variant, language))

        learned_rows, learned_summary = run_learned_oof(query_items, strong_by_qid, constraints_by_qid, language, args.seed, args.folds, args.dev_candidates)
        variants["metric_learned_rerank_oof"] = learned_rows
        write_jsonl(args.out_dir / f"{language}_metric_learned_rerank_oof_predictions.jsonl", learned_rows)
        learned_metrics = aggregate(learned_rows)
        metric_rows.append({"language": language, "run": "metric_learned_rerank_oof", **learned_metrics})
        delta_rows.append({"language": language, "run": "metric_learned_rerank_oof", "baseline": "recomputed_strong_full_dataset", **{f"delta_{m}": learned_metrics[m] - strong_metrics[m] for m in METRICS}})
        subset_rows.extend(subset_metrics(learned_rows, "metric_learned_rerank_oof", language))
        split_summary["languages"][language] = learned_summary
        (args.out_dir / f"{language}_learned_oof_split_and_params.json").write_text(json.dumps(learned_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_all.extend(audit_rows(query_items, constraints_by_qid, language))
        all_case_variants.update({f"{language}_{name}": rows for name, rows in variants.items()})

        qa_metrics_path = args.query_aware_confirmatory_dir / "metrics_summary.csv"
        if qa_metrics_path.exists():
            with qa_metrics_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row["language"] == language and row["run"] == "fixed_confirmatory_calibration_test_split":
                        metric_rows.append({"language": language, "run": "previous_generic_confirmatory_test_split", **{m: float(row[m]) for m in METRICS}})

        ocr_metrics_path = args.ocr_rerun_dir / f"{language}_query_router_multi_granularity_mono_metrics.json"
        if ocr_metrics_path.exists():
            ocr_metrics = json.loads(ocr_metrics_path.read_text(encoding="utf-8"))
            metric_rows.append({"language": language, "run": "previous_naive_ocr_pdf_router_mono", **{m: ocr_metrics[m] for m in METRICS}})

    write_csv(args.out_dir / "metrics_summary.csv", metric_rows, ["language", "run", *METRICS])
    write_csv(args.out_dir / "delta_vs_strong.csv", delta_rows, ["language", "run", "baseline", *[f"delta_{m}" for m in METRICS]])
    write_csv(args.out_dir / "subset_metrics_by_query_type.csv", subset_rows, ["language", "run", "query_type", "query_count", *METRICS])
    write_csv(
        args.out_dir / "financial_feature_audit.csv",
        audit_all,
        [
            "language",
            "query_id",
            "query_type",
            "canonical_metrics",
            "metric_terms",
            "year_terms",
            "unit_terms",
            "statement_type",
            "top10_avg_metric_match",
            "top10_avg_year_match",
            "top10_avg_statement_match",
            "top10_avg_table_density",
            "top10_avg_cooccurrence",
            "top10_avg_numeric_context",
        ],
    )
    (args.out_dir / "learned_oof_split_summary.json").write_text(json.dumps(split_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    (args.out_dir / "model_downloads.md").write_text(
        "# Model / Data Downloads\n\n"
        "No new model, dataset, dictionary, or dependency was downloaded for this run.\n\n"
        "The financial metric dictionary and alias map are implemented locally in `scripts/run_financial_metric_aware_reranking.py` and were used in the final experiment.\n",
        encoding="utf-8",
    )
    (args.out_dir / "reproduction_commands.md").write_text(
        "# Reproduction Commands\n\n"
        "```bash\n"
        "cd <repository-root>\n"
        "RUN_DIR=artifacts/retrieval/financial_metric_aware_reranking_$(date +%Y%m%d_%H%M%S)\n"
        "mkdir -p \"$RUN_DIR/logs\"\n"
        "conda run -n finragbench python scripts/run_financial_metric_aware_reranking.py \\\n"
        "  --strong-base-dir artifacts/retrieval/rerun_tab_safe_20260624_074152 \\\n"
        "  --ocr-rerun-dir artifacts/retrieval/real_router_multigranularity_20260624_131957/candidate_ocr_pdf_merged_rerun \\\n"
        "  --index-dir artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged \\\n"
        "  --query-aware-confirmatory-dir artifacts/retrieval/query_aware_calibration_confirmatory_20260630_072446 \\\n"
        "  --out-dir \"$RUN_DIR\" --seed 42 --folds 5 --dev-candidates 25\n"
        "```\n",
        encoding="utf-8",
    )

    strong_rows_for_cases = []
    for language in ("en", "zh"):
        strong_by_qid, _, constraints_by_qid, _ = build_language_inputs(args, language)
        strong_rows_for_cases.extend(strong_rows_from_inputs(strong_by_qid, constraints_by_qid))
    write_case_studies(args.out_dir / "case_studies.md", strong_rows_for_cases, all_case_variants)
    print(json.dumps({"out_dir": str(args.out_dir), "metrics_rows": metric_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
