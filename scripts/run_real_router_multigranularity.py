#!/usr/bin/env python3
"""Run real page/text/block multi-granularity retrieval fusion.

This script uses existing ColQwen2 and MonoQwen2-VL runs as visual baselines,
plus BM25 indexes built from real PDF page text and PDF text-fragment blocks.
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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = ("nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10")
EN_TOKEN_RE = re.compile(r"[a-z0-9]+")
ZH_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


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


def tokenize(text: str, language: str) -> list[str]:
    text = "" if text is None else str(text)
    text = text.lower()
    if language == "zh":
        chars = ZH_CJK_RE.findall(text)
        bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        alnum = ALNUM_RE.findall(text)
        return chars + bigrams + alnum
    return EN_TOKEN_RE.findall(text)


@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_lens: list[int]
    postings: dict[str, list[tuple[int, int]]]
    avgdl: float
    total_docs: int
    meta: dict[str, dict[str, Any]]
    language: str

    @classmethod
    def build(cls, rows: list[dict[str, Any]], doc_id_key: str, text_key: str, language: str) -> "BM25Index":
        doc_ids: list[str] = []
        doc_lens: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        meta: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc_id = row[doc_id_key]
            tokens = tokenize(row.get(text_key, ""), language)
            if not tokens:
                continue
            idx = len(doc_ids)
            doc_ids.append(doc_id)
            doc_lens.append(len(tokens))
            meta[doc_id] = row
            for term, tf in Counter(tokens).items():
                postings[term].append((idx, tf))
        avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
        return cls(doc_ids, doc_lens, dict(postings), avgdl, len(doc_ids), meta, language)

    def search(self, query: str, topk: int) -> list[tuple[str, float]]:
        q_terms = Counter(tokenize(query, self.language))
        if not q_terms or not self.total_docs:
            return []
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.2
        b = 0.75
        for term, qtf in q_terms.items():
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in plist:
                denom = tf + k1 * (1.0 - b + b * self.doc_lens[doc_idx] / max(self.avgdl, 1e-9))
                scores[doc_idx] += idf * (tf * (k1 + 1.0) / denom) * (1.0 + 0.1 * math.log1p(qtf))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))[:topk]
        return [(self.doc_ids[idx], score) for idx, score in ranked]


def normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {key: 1.0 for key in score_map}
    return {key: (value - lo) / (hi - lo) for key, value in score_map.items()}


def query_type_and_weights(query: str, language: str, include_mono: bool = False) -> tuple[str, dict[str, float]]:
    q = query.lower()
    has_calc = any(token in q for token in ["calculate", "change", "average", "ratio", "difference", "sum", "total", "increase", "decrease", "计算", "变化", "平均", "合计", "增长", "减少"])
    has_table = any(token in q for token in ["table", "statement", "balance sheet", "income", "cash flow", "表", "报表", "资产负债", "利润", "现金流"])
    has_chart = any(token in q for token in ["chart", "figure", "trend", "图", "趋势"])
    if has_chart:
        qtype = "chart_or_figure"
        weights = {"visual": 0.54, "text": 0.18, "block": 0.28}
    elif has_calc or has_table:
        qtype = "financial_table_or_calculation"
        weights = {"visual": 0.42, "text": 0.24, "block": 0.34}
    else:
        qtype = "text_fact"
        weights = {"visual": 0.38, "text": 0.42, "block": 0.20}
    if language == "zh":
        weights = {**weights, "text": weights["text"] + 0.03, "visual": weights["visual"] - 0.03}
    if include_mono:
        weights = {k: v * 0.78 for k, v in weights.items()}
        weights["mono"] = 0.22
    total = sum(weights.values())
    return qtype, {key: value / total for key, value in weights.items()}


def static_weights(include_mono: bool = False) -> dict[str, float]:
    weights = {"visual": 0.40, "text": 0.35, "block": 0.25}
    if include_mono:
        weights = {k: v * 0.78 for k, v in weights.items()}
        weights["mono"] = 0.22
    return weights


def rank_to_score(candidates: list[dict[str, Any]], score_key: str = "score") -> dict[str, float]:
    return {cand["corpus_id"]: float(cand.get(score_key, cand.get("score", 0.0))) for cand in candidates}


def block_scores_to_pages(block_hits: list[tuple[str, float]], block_index: BM25Index, max_blocks_per_page: int = 3) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    page_scores: dict[str, float] = defaultdict(float)
    contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block_id, score in block_hits:
        row = block_index.meta.get(block_id)
        if not row:
            continue
        cid = row["corpus_id"]
        page_scores[cid] = max(page_scores[cid], score)
        if len(contributions[cid]) < max_blocks_per_page:
            contributions[cid].append(
                {
                    "block_id": block_id,
                    "score": score,
                    "bbox": row.get("bbox"),
                    "text": str(row.get("text", ""))[:300],
                    "source": row.get("source"),
                }
            )
    return dict(page_scores), dict(contributions)


def evaluate(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> dict[str, float]:
    rels = [1 if cand["corpus_id"] in qrels and qrels[cand["corpus_id"]] > 0 else 0 for cand in candidates]
    total_rel = max(1, sum(1 for value in qrels.values() if value > 0))
    out: dict[str, float] = {}
    for k in (5, 10):
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels[:k]))
        ideal = sum(1.0 / math.log2(idx + 2) for idx in range(min(total_rel, k)))
        out[f"nDCG@{k}"] = dcg / ideal if ideal else 0.0
        out[f"Recall@{k}"] = sum(rels[:k]) / total_rel
    rr = 0.0
    for idx, rel in enumerate(rels[:10], start=1):
        if rel:
            rr = 1.0 / idx
            break
    out["MRR@10"] = rr
    return out


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: sum(row["metrics"][metric] for row in rows) / len(rows) for metric in METRICS}


def load_strong_metrics(base_dir: Path, language: str) -> dict[str, Any]:
    path = base_dir / f"colqwen2_{language}_monoqwen2vl_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def first_rank(candidates: list[dict[str, Any]], qrels: dict[str, int]) -> int | None:
    for idx, cand in enumerate(candidates, start=1):
        if cand["corpus_id"] in qrels and qrels[cand["corpus_id"]] > 0:
            return idx
    return None


def run_language(language: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    prefix = "en" if language == "en" else "zh"
    baseline_rows = read_jsonl(args.strong_base_dir / f"colqwen2_{prefix}_predictions.jsonl")
    strong_rows = read_jsonl(args.strong_base_dir / f"colqwen2_{prefix}_monoqwen2vl_predictions_raw.jsonl")
    strong_by_qid = {row["query_id"]: row for row in strong_rows}
    page_rows = read_jsonl(args.index_dir / f"{prefix}_pages.jsonl")
    block_rows = read_jsonl(args.index_dir / f"{prefix}_blocks.jsonl")
    page_index = BM25Index.build(page_rows, "corpus_id", "text", language)
    block_index = BM25Index.build(block_rows, "block_id", "text", language)

    stats = {
        "language": language,
        "queries": len(baseline_rows),
        "page_index_docs": page_index.total_docs,
        "block_index_docs": block_index.total_docs,
    }
    (out_dir / f"{prefix}_bm25_index_runtime_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    variants = {
        "multi_granularity_index": {"router": False, "mono": False},
        "query_router_multi_granularity": {"router": True, "mono": False},
        "query_router_multi_granularity_mono": {"router": True, "mono": True},
    }
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in variants}
    change_rows = []
    contribution_counter: dict[str, Counter[str]] = {name: Counter() for name in variants}
    query_type_counter: Counter[str] = Counter()

    for row_idx, row in enumerate(baseline_rows, start=1):
        query = row.get("query") or ""
        qid = row["query_id"]
        qrels = row.get("qrels", {})
        visual_scores = normalize(rank_to_score(row["candidates"]))
        text_scores = normalize(dict(page_index.search(query, args.text_topk)))
        block_page_raw, block_contrib = block_scores_to_pages(block_index.search(query, args.block_topk), block_index)
        block_scores = normalize(block_page_raw)
        strong_row = strong_by_qid.get(qid)
        mono_scores = normalize(rank_to_score(strong_row.get("candidates", []), "rerank_score") if strong_row else {})
        baseline_rank = first_rank(row["candidates"], qrels)
        strong_rank = first_rank(strong_row.get("candidates", []) if strong_row else [], qrels)

        for variant_name, config in variants.items():
            if config["router"]:
                qtype, weights = query_type_and_weights(query, language, include_mono=config["mono"])
            else:
                qtype, weights = "static_fusion", static_weights(include_mono=config["mono"])
            query_type_counter[qtype] += 1
            all_ids = set(visual_scores) | set(text_scores) | set(block_scores)
            if config["mono"]:
                all_ids |= set(mono_scores)
            fused: list[dict[str, Any]] = []
            for cid in all_ids:
                source_scores = {
                    "visual": visual_scores.get(cid, 0.0),
                    "text": text_scores.get(cid, 0.0),
                    "block": block_scores.get(cid, 0.0),
                }
                if config["mono"]:
                    source_scores["mono"] = mono_scores.get(cid, 0.0)
                final_score = sum(weights.get(name, 0.0) * source_scores.get(name, 0.0) for name in weights)
                if final_score <= 0:
                    continue
                best_source = max(source_scores.items(), key=lambda kv: kv[1])[0]
                contribution_counter[variant_name][best_source] += 1
                fused.append(
                    {
                        "corpus_id": cid,
                        "score": final_score,
                        "source_scores": source_scores,
                        "block_contributions": block_contrib.get(cid, [])[:2],
                    }
                )
            fused.sort(key=lambda item: (-item["score"], item["corpus_id"]))
            candidates = [
                {
                    "corpus_id": cand["corpus_id"],
                    "rank": idx,
                    "score": cand["score"],
                    "run_name": f"real_{variant_name}_{prefix}",
                    "source_scores": cand["source_scores"],
                    "block_contributions": cand["block_contributions"],
                }
                for idx, cand in enumerate(fused[: args.output_topk], start=1)
            ]
            metrics = evaluate(candidates, qrels)
            out_row = {
                "query_id": qid,
                "query": query,
                "answer": row.get("answer"),
                "qrels": qrels,
                "query_type": qtype,
                "router_weights": weights,
                "metrics": metrics,
                "candidates": candidates,
            }
            outputs[variant_name].append(out_row)
            if variant_name == "query_router_multi_granularity_mono":
                change_rows.append(
                    {
                        "query_id": qid,
                        "query_type": qtype,
                        "baseline_gold_rank": baseline_rank,
                        "strong_gold_rank": strong_rank,
                        "real_gold_rank": first_rank(candidates, qrels),
                        "top10_changed_vs_strong": [c["corpus_id"] for c in candidates[:10]]
                        != [c["corpus_id"] for c in (strong_row.get("candidates", []) if strong_row else [])[:10]],
                    }
                )
        if row_idx % 100 == 0:
            print(f"{prefix}: processed {row_idx}/{len(baseline_rows)} queries", flush=True)

    metrics_summary = {}
    for variant_name, rows in outputs.items():
        write_jsonl(out_dir / f"{prefix}_{variant_name}_predictions.jsonl", rows)
        metrics_summary[variant_name] = aggregate_metrics(rows)
        (out_dir / f"{prefix}_{variant_name}_metrics.json").write_text(
            json.dumps(metrics_summary[variant_name], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (out_dir / f"{prefix}_ranking_change_summary.json").write_text(json.dumps(change_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_query_type_distribution.json").write_text(json.dumps(query_type_counter, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_granularity_contribution_counts.json").write_text(
        json.dumps({name: dict(counter) for name, counter in contribution_counter.items()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "language": language,
        "runtime_stats": stats,
        "metrics": metrics_summary,
        "strong_baseline": load_strong_metrics(args.strong_base_dir, prefix),
    }


def write_summary(out_dir: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    rows = []
    for lang, lang_summary in summary.items():
        strong = lang_summary["strong_baseline"]
        rows.append([lang, "strong_colqwen2_monoqwen2vl", *[strong[m] for m in METRICS]])
        for variant, metrics in lang_summary["metrics"].items():
            rows.append([lang, variant, *[metrics[m] for m in METRICS]])
    with (out_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "run", *METRICS])
        writer.writerows(rows)
    lines = ["# Real Router Multi-Granularity Metrics", "", "| Language | Run | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 | MRR@10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lang, run, *vals = row
        lines.append("| " + " | ".join([lang, run, *[f"{value:.6f}" for value in vals]]) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `innovation_router_index_20260624_123649` is treated only as a proxy prototype / quick ablation.",
            "- This run uses real PDF-native page text and PDF text-fragment region blocks; no filename/page-number/ranking proxies are used for text or block scores.",
            "- Official OCR/table/chart metadata was not found during audit, so this run is not an image-OCR/table-structure experiment.",
        ]
    )
    (out_dir / "final_metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = {
        "index_dir": str(args.index_dir),
        "strong_base_dir": str(args.strong_base_dir),
        "text_topk": args.text_topk,
        "block_topk": args.block_topk,
        "output_topk": args.output_topk,
    }
    (out_dir / "real_router_multigranularity_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strong-base-dir", type=Path, default=Path("artifacts/retrieval/rerun_tab_safe_20260624_074152"))
    parser.add_argument("--text-topk", type=int, default=300)
    parser.add_argument("--block-topk", type=int, default=300)
    parser.add_argument("--output-topk", type=int, default=100)
    parser.add_argument("--languages", nargs="+", default=["en", "zh"], choices=["en", "zh"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for language in args.languages:
        summary[language] = run_language(language, args, args.out_dir)
    write_summary(args.out_dir, summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
