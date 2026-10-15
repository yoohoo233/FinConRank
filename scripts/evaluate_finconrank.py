#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from finconrank.common import METRICS, aggregate_metrics, evaluate, normalize_scores, read_jsonl, write_jsonl
from run_financial_metric_aware_reranking import candidate_weights, page_financial_features


def load_evidence(index_dir: Path, language: str, wanted: set[str]) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    text, title = {}, {}
    with (index_dir / f"{language}_pages.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cid = row["corpus_id"]
            if cid in wanted:
                title[cid] = str(row.get("title", "") or "")
                text[cid] = f"{row.get('title', '')} {row.get('text', '')}".strip()
    blocks: dict[str, list[str]] = defaultdict(list)
    with (index_dir / f"{language}_blocks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cid = row["corpus_id"]
            if cid in wanted and len(blocks[cid]) < 80 and row.get("text"):
                blocks[cid].append(str(row["text"]))
    return text, title, blocks


def load_source_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for row in read_jsonl(path):
        output[row["query_id"]] = {candidate["corpus_id"]: candidate.get("source_scores", {}) for candidate in row.get("candidates", [])}
    return output


def feature_rows(
    row: dict[str, Any],
    constraints: dict[str, Any],
    page_text: dict[str, str],
    page_title: dict[str, str],
    block_text: dict[str, list[str]],
    source_scores: dict[str, dict[str, float]],
    mono_field: str,
) -> list[dict[str, Any]]:
    candidates = row.get("candidates", [])
    visual = normalize_scores(candidates, "score")
    mono = normalize_scores(candidates, mono_field)
    output = []
    for rank, candidate in enumerate(candidates, start=1):
        cid = candidate["corpus_id"]
        financial = page_financial_features(page_text.get(cid, ""), page_title.get(cid, ""), block_text.get(cid, []), constraints)
        source = source_scores.get(cid, {})
        output.append({
            "corpus_id": cid,
            "strong_rank": rank,
            "rank_prior": 1.0 / math.log2(rank + 1.0),
            "mono": mono.get(cid, 0.0),
            "visual": visual.get(cid, 0.0),
            "text": float(source.get("text", 0.0) or 0.0),
            "block": float(source.get("block", 0.0) or 0.0),
            **financial,
            "source_candidate": candidate,
        })
    return output


def metric_vector(item: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        item["rank_prior"], item["mono"], item["visual"], item["text"], item["block"],
        item["metric_match_score"], item["year_match_score"], item["statement_match_score"],
        item["unit_match_score"], item["table_density_score"], item["cooccurrence_score"], item["numeric_context_score"],
    ], dtype=float)


def rank_from_items(items: list[dict[str, Any]], scores: list[float], run_name: str) -> list[dict[str, Any]]:
    ranked = sorted(zip(items, scores), key=lambda pair: (pair[1], -pair[0]["strong_rank"]), reverse=True)
    output = []
    for rank, (item, score) in enumerate(ranked, start=1):
        candidate = dict(item["source_candidate"])
        candidate["rank"] = rank
        candidate["metric_aware_score"] = float(score)
        candidate["run_name"] = run_name
        output.append(candidate)
    return output


def mean_ndcg5(qids: list[str], items: dict[str, list[dict[str, Any]]], rows: dict[str, dict[str, Any]], weights: np.ndarray) -> float:
    values = []
    for qid in qids:
        candidates = rank_from_items(items[qid], [float(weights @ metric_vector(item)) for item in items[qid]], "metric_fit")
        values.append(evaluate(candidates, rows[qid].get("qrels", {}))["nDCG@5"])
    return sum(values) / len(values) if values else 0.0


def fit_metric_weights(items: dict[str, list[dict[str, Any]]], rows: dict[str, dict[str, Any]], split: dict[str, str], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    valid = [qid for qid, values in items.items() if values]
    train = [qid for qid in valid if split[qid] == "train"]
    dev = [qid for qid in valid if split[qid] == "dev"]
    weights = candidate_weights(12, seed)
    train_scores = sorted([(mean_ndcg5(train, items, rows, weight), idx, weight) for idx, weight in enumerate(weights)], reverse=True, key=lambda row: row[0])
    shortlist = train_scores[:12]
    dev_scores = sorted([(mean_ndcg5(dev, items, rows, weight), train_score, idx, weight) for train_score, idx, weight in shortlist], reverse=True, key=lambda row: (row[0], row[1]))
    dev_score, train_score, index, selected = dev_scores[0]
    return selected, {"weight_index": index, "weights": selected.tolist(), "train_nDCG@5": train_score, "dev_nDCG@5": dev_score, "train_queries": len(train), "dev_queries": len(dev)}


def materialize(rows: dict[str, dict[str, Any]], candidates: dict[str, list[dict[str, Any]]], split: dict[str, str], run_name: str, query_types: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for qid, source in rows.items():
        ranked = candidates.get(qid, [])
        output.append({
            "query_id": qid,
            "query": source.get("query"),
            "answer": source.get("answer"),
            "qrels": source.get("qrels", {}),
            "split": split[qid],
            "query_type": query_types[qid],
            "metrics": evaluate(ranked, source.get("qrels", {})),
            "candidates": ranked,
            "run_name": run_name,
        })
    return output


def direct_rank(row: dict[str, Any], field: str, run_name: str, depth: int) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in row.get("candidates", [])[:depth]]
    candidates.sort(key=lambda item: (float(item.get(field, -1e9) if item.get(field) is not None else -1e9), -int(item.get("rank", 999))), reverse=True)
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["run_name"] = run_name
    return candidates


def fusion_grid() -> list[np.ndarray]:
    output = []
    for parts in itertools.product(range(0, 11, 2), repeat=4):
        if sum(parts) == 10 and parts[2] > 0:
            output.append(np.asarray(parts, dtype=float) / 10.0)
    return output


def fit_fusion(
    feature_by_qid: dict[str, dict[str, tuple[dict[str, Any], np.ndarray]]],
    rows: dict[str, dict[str, Any]],
    split: dict[str, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    def score(qids: list[str], weight: np.ndarray) -> float:
        values = []
        for qid in qids:
            ranked = sorted(feature_by_qid[qid].values(), key=lambda pair: float(weight @ pair[1]), reverse=True)
            values.append(evaluate([pair[0] for pair in ranked], rows[qid].get("qrels", {}))["nDCG@5"])
        return sum(values) / len(values) if values else 0.0
    qids = list(feature_by_qid)
    train = [qid for qid in qids if split[qid] == "train"]
    dev = [qid for qid in qids if split[qid] == "dev"]
    candidates = fusion_grid()
    train_scores = sorted([(score(train, weight), idx, weight) for idx, weight in enumerate(candidates)], reverse=True, key=lambda row: row[0])
    shortlist = train_scores[:10]
    dev_scores = sorted([(score(dev, weight), train_score, idx, weight) for train_score, idx, weight in shortlist], reverse=True, key=lambda row: (row[0], row[1]))
    dev_score, train_score, index, selected = dev_scores[0]
    return selected, {"weight_index": index, "weights_colq_mono_fincon_metric": selected.tolist(), "train_nDCG@5": train_score, "dev_nDCG@5": dev_score}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/retrieval/rerun_tab_safe_20260624_074152"))
    parser.add_argument("--mono-top50-dir", type=Path, required=True)
    parser.add_argument("--fincon-colq-dir", type=Path, required=True)
    parser.add_argument("--fincon-mono-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged"))
    parser.add_argument("--router-dir", type=Path, default=Path("artifacts/retrieval/real_router_multigranularity_20260624_131957/candidate_ocr_pdf_merged_rerun"))
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--depth-en", type=int)
    parser.add_argument("--depth-zh", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_manifest = json.loads((args.run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    split = {qid: item["split"] for qid, item in split_manifest["queries"].items()}
    constraint_rows = {row["query_id"]: row for row in read_jsonl(args.run_dir / "query_constraints.jsonl")}
    summary_rows, delta_rows, subset_rows = [], [], []
    final_selection = {}

    for language in ("en", "zh"):
        depth = (args.depth_en if language == "en" else args.depth_zh) or args.depth
        col_rows = read_jsonl(args.baseline_dir / f"colqwen2_{language}_predictions.jsonl")
        mono_rows = read_jsonl(args.mono_top50_dir / f"{language}_monoqwen2vl_top50_predictions.jsonl")
        fincon_col_rows = read_jsonl(args.fincon_colq_dir / f"{language}_finconrank_colq_predictions.jsonl")
        fincon_mono_rows = read_jsonl(args.fincon_mono_dir / f"{language}_finconrank_mono_predictions.jsonl")
        rows = {row["query_id"]: row for row in col_rows}
        mono = {row["query_id"]: row for row in mono_rows}
        fin_col = {row["query_id"]: row for row in fincon_col_rows}
        fin_mono = {row["query_id"]: row for row in fincon_mono_rows}
        query_types = {qid: constraint_rows[qid]["query_type"] for qid in rows}
        wanted = {candidate["corpus_id"] for row in col_rows for candidate in row.get("candidates", [])[: max(depth, 50)]}
        page_text, page_title, blocks = load_evidence(args.index_dir, language, wanted)
        source_scores = load_source_scores(args.router_dir / f"{language}_query_router_multi_granularity_mono_predictions.jsonl")

        run_candidates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for qid, row in rows.items():
            col = [dict(item) for item in row.get("candidates", [])[:depth]]
            for rank, item in enumerate(col, start=1):
                item["rank"] = rank
            run_candidates["colqwen2"][qid] = col
            run_candidates["colqwen2_monoqwen2vl"][qid] = direct_rank(mono[qid], "rerank_score", "colqwen2_monoqwen2vl", depth)
            run_candidates["colqwen2_finconrank"][qid] = direct_rank(fin_col[qid], "finconrank_score", "colqwen2_finconrank", depth)
            run_candidates["colqwen2_monoqwen2vl_finconrank"][qid] = direct_rank(fin_mono[qid], "finconrank_score", "colqwen2_monoqwen2vl_finconrank", depth)

        metric_sources = {
            "colqwen2_monoqwen2vl_metric_aware": (run_candidates["colqwen2_monoqwen2vl"], "rerank_score"),
            "colqwen2_finconrank_metric_aware": (run_candidates["colqwen2_finconrank"], "finconrank_score"),
            "colqwen2_monoqwen2vl_finconrank_metric_aware": (run_candidates["colqwen2_monoqwen2vl_finconrank"], "finconrank_score"),
        }
        metric_params = {}
        metric_scores_by_mode: dict[str, dict[str, dict[str, float]]] = {}
        for mode, (candidate_map, score_field) in metric_sources.items():
            feature_by_qid = {}
            synthetic_rows = {}
            for qid, candidates in candidate_map.items():
                synthetic = dict(rows[qid])
                synthetic["candidates"] = candidates
                synthetic_rows[qid] = synthetic
                feature_by_qid[qid] = feature_rows(synthetic, constraint_rows[qid]["constraints"], page_text, page_title, blocks, source_scores.get(qid, {}), score_field)
            weights, params = fit_metric_weights(feature_by_qid, synthetic_rows, split, args.seed + (0 if language == "en" else 1000))
            metric_params[mode] = params
            metric_scores_by_mode[mode] = {}
            for qid, items in feature_by_qid.items():
                scores = [float(weights @ metric_vector(item)) for item in items]
                ranked = rank_from_items(items, scores, mode)
                run_candidates[mode][qid] = ranked
                metric_scores_by_mode[mode][qid] = {item["corpus_id"]: float(item["metric_aware_score"]) for item in ranked}

        fusion_features = {}
        for qid in rows:
            col_map = {item["corpus_id"]: item for item in run_candidates["colqwen2"][qid]}
            mono_norm = normalize_scores(run_candidates["colqwen2_monoqwen2vl"][qid], "rerank_score")
            fin_norm = normalize_scores(run_candidates["colqwen2_finconrank"][qid], "finconrank_score")
            metric_map = metric_scores_by_mode["colqwen2_finconrank_metric_aware"][qid]
            metric_values = list(metric_map.values())
            low, high = (min(metric_values), max(metric_values)) if metric_values else (0.0, 0.0)
            fusion_features[qid] = {}
            col_norm = normalize_scores(list(col_map.values()), "score")
            for cid, candidate in col_map.items():
                metric_norm = (metric_map.get(cid, low) - low) / (high - low) if high > low else 0.5
                fusion_features[qid][cid] = (dict(candidate), np.asarray([col_norm.get(cid, 0.0), mono_norm.get(cid, 0.0), fin_norm.get(cid, 0.0), metric_norm], dtype=float))
        fusion_weights, fusion_params = fit_fusion(fusion_features, rows, split)
        for qid, feature_map in fusion_features.items():
            ordered = sorted(feature_map.values(), key=lambda pair: float(fusion_weights @ pair[1]), reverse=True)
            candidates = []
            for rank, (candidate, vector) in enumerate(ordered, start=1):
                candidate["rank"] = rank
                candidate["fusion_score"] = float(fusion_weights @ vector)
                candidate["run_name"] = "finconrank_full_fusion"
                candidates.append(candidate)
            run_candidates["finconrank_full_fusion"][qid] = candidates

        dev_methods = {}
        for mode in ("colqwen2_finconrank_metric_aware", "colqwen2_monoqwen2vl_finconrank_metric_aware", "finconrank_full_fusion"):
            material = materialize(rows, run_candidates[mode], split, mode, query_types)
            dev_methods[mode] = aggregate_metrics(row for row in material if row["split"] == "dev")["nDCG@5"]
        selected_mode = max(dev_methods, key=dev_methods.get)
        final_selection[language] = {"selected_on_dev": selected_mode, "candidate_depth": depth, "dev_nDCG@5": dev_methods, "metric_params": metric_params, "fusion_params": fusion_params}

        material_by_mode = {}
        for mode, candidate_map in run_candidates.items():
            material = materialize(rows, candidate_map, split, mode, query_types)
            material_by_mode[mode] = material
            write_jsonl(args.run_dir / f"{language}_{mode}_predictions.jsonl", material)
            for evaluation_split in ("dev", "test"):
                selected = [row for row in material if row["split"] == evaluation_split]
                metrics = aggregate_metrics(selected)
                summary_rows.append({"language": language, "evaluation_split": evaluation_split, "run": mode, "query_count": len(selected), **metrics})

        baseline_mode = "colqwen2_monoqwen2vl"
        test_baseline = next(row for row in summary_rows if row["language"] == language and row["evaluation_split"] == "test" and row["run"] == baseline_mode)
        for row in [item for item in summary_rows if item["language"] == language and item["evaluation_split"] == "test"]:
            delta_rows.append({"language": language, "run": row["run"], "baseline": baseline_mode, **{f"delta_{metric}": row[metric] - test_baseline[metric] for metric in METRICS}})
        for mode, material in material_by_mode.items():
            for query_type in sorted(set(query_types.values())):
                selected = [row for row in material if row["split"] == "test" and row["query_type"] == query_type]
                if selected:
                    subset_rows.append({"language": language, "run": mode, "query_type": query_type, "query_count": len(selected), **aggregate_metrics(selected)})

    write_csv(args.run_dir / "metrics_summary.csv", summary_rows)
    write_csv(args.run_dir / "delta_vs_same_split_baselines.csv", delta_rows)
    write_csv(args.run_dir / "query_type_metrics.csv", subset_rows)
    (args.run_dir / "deployment_selection.json").write_text(json.dumps(final_selection, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
