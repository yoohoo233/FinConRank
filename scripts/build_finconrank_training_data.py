#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finconrank.common import page_document_id, read_jsonl, write_jsonl
from finconrank.constraints import METRICS, PERIODS, SCOPES, STATEMENTS, UNITS, YEAR_RE, find_terms


NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\(?\d[\d,.]*(?:\.\d+)?\)?%?")
ALL_METRIC_TERMS = sorted({alias for meta in METRICS.values() for alias in meta["aliases"]}, key=len, reverse=True)
ALL_STATEMENT_TERMS = sorted({term for terms in STATEMENTS.values() for term in terms}, key=len, reverse=True)
ANY_METRIC_RE = re.compile("|".join(re.escape(term.lower()) for term in ALL_METRIC_TERMS), flags=re.IGNORECASE)
ANY_STATEMENT_RE = re.compile("|".join(re.escape(term.lower()) for term in ALL_STATEMENT_TERMS), flags=re.IGNORECASE)
ANY_PERIOD_RE = re.compile("|".join(re.escape(term.lower()) for term in PERIODS), flags=re.IGNORECASE)
ANY_UNIT_RE = re.compile("|".join(re.escape(term.lower()) for term in UNITS), flags=re.IGNORECASE)
ANY_SCOPE_RE = re.compile("|".join(re.escape(term.lower()) for term in SCOPES), flags=re.IGNORECASE)


def load_selected_text(path: Path, wanted: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cid = row["corpus_id"]
            if cid in wanted:
                output[cid] = f"{row.get('title', '')}\n{row.get('text', '')}".lower()
    return output


def evidence_dimensions(text: str, constraints: dict[str, Any], candidate_doc: str = "") -> dict[str, bool]:
    metric_terms = list(constraints.get("metric_terms", []))
    for metric in constraints.get("canonical_metrics", []):
        metric_terms.extend(METRICS[metric]["aliases"])
    statement = constraints.get("statement_type", "unknown")
    entity = constraints.get("entity")
    entity_tokens = [token.lower() for token in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", entity or "")]
    page_years = set(YEAR_RE.findall(text))
    return {
        "metric": not metric_terms or bool(find_terms(text, metric_terms)),
        "year": not constraints.get("year_terms") or set(constraints["year_terms"]).issubset(page_years),
        "entity": not entity_tokens or any(token in text for token in entity_tokens),
        "period": not constraints.get("period_terms") or bool(find_terms(text, constraints["period_terms"])),
        "unit": not constraints.get("unit_terms") or bool(find_terms(text, constraints["unit_terms"])),
        "statement": statement == "unknown" or bool(find_terms(text, STATEMENTS.get(statement, []))),
        "scope": not constraints.get("accounting_scope") or bool(find_terms(text, constraints["accounting_scope"])),
        "numeric": bool(NUM_RE.search(text)),
        "text_available": len(text.strip()) >= 80,
        "has_any_year": bool(page_years),
        "has_any_metric": bool(ANY_METRIC_RE.search(text)),
        "has_any_statement": bool(ANY_STATEMENT_RE.search(text)),
        "has_any_period": bool(ANY_PERIOD_RE.search(text)),
        "has_any_unit": bool(ANY_UNIT_RE.search(text)),
        "has_any_scope": bool(ANY_SCOPE_RE.search(text)),
        "entity_conflict": bool(entity_tokens) and not any(token in candidate_doc.lower() for token in entity_tokens),
    }


def classify_negative(
    candidate: dict[str, Any],
    text: str,
    constraints: dict[str, Any],
    gold_docs: set[str],
    rank: int,
) -> tuple[list[str], dict[str, bool]]:
    candidate_doc = page_document_id(candidate["corpus_id"])
    dims = evidence_dimensions(text, constraints, candidate_doc)
    types = []
    if candidate_doc in gold_docs:
        types.append("same_document_wrong_page")
    if rank <= 10:
        types.append("visually_similar_negative")
    required = {
        "metric": bool(constraints.get("canonical_metrics") or constraints.get("metric_terms")),
        "year": bool(constraints.get("year_terms")),
        "entity": bool(constraints.get("entity") and constraints.get("entity_confidence", 0) >= 0.8),
        "period": bool(constraints.get("period_terms")),
        "unit": bool(constraints.get("unit_terms")),
        "statement": constraints.get("statement_type") != "unknown",
        "scope": bool(constraints.get("accounting_scope")),
    }
    conflict_map = {
        "entity": "wrong_company_entity",
        "period": "wrong_reporting_period",
        "unit": "wrong_unit_currency",
        "statement": "wrong_statement_type",
        "scope": "wrong_accounting_scope",
    }
    misses = [name for name, needed in required.items() if needed and not dims[name]]
    for miss in misses:
        if miss == "year":
            if dims["has_any_year"]:
                types.append("same_metric_wrong_year" if required["metric"] and dims["metric"] else "wrong_year")
        elif miss == "metric":
            if dims["has_any_metric"]:
                types.append("same_year_wrong_metric" if required["year"] and dims["year"] else "wrong_metric")
        elif miss == "statement":
            if dims["has_any_statement"]:
                types.append(conflict_map[miss])
        elif miss == "period":
            if dims["has_any_period"]:
                types.append(conflict_map[miss])
        elif miss == "unit":
            if dims["has_any_unit"]:
                types.append(conflict_map[miss])
        elif miss == "scope":
            if dims["has_any_scope"]:
                types.append(conflict_map[miss])
        elif miss == "entity":
            if dims["entity_conflict"]:
                types.append(conflict_map[miss])
    required_count = sum(required.values())
    explicit_conflict = any(name.startswith("wrong_") or name in {"same_metric_wrong_year", "same_year_wrong_metric"} for name in types)
    if required_count >= 2 and len(misses) == 1 and explicit_conflict:
        types.append("single_constraint_counterfactual")
    if required_count == 0 and rank <= 10 and candidate_doc not in gold_docs:
        types.append("constraint_free_visual_negative")
    if rank >= 60 and candidate_doc not in gold_docs:
        types.append("random_easy_negative")
    return sorted(set(types)), dims


def verified_status(types: list[str], dims: dict[str, bool], candidate_doc: str, gold_docs: set[str], qrel_known_negative: bool) -> tuple[str, str]:
    if not qrel_known_negative:
        return "reject", "candidate is a known positive"
    informative_conflicts = [
        name for name in types
        if name.startswith("wrong_") or name in {"same_metric_wrong_year", "same_year_wrong_metric", "single_constraint_counterfactual"}
    ]
    all_core_match = all(dims[name] for name in ("metric", "year", "entity", "period", "unit", "statement", "scope"))
    if candidate_doc in gold_docs and all_core_match:
        return "uncertain", "same-report page satisfies all observable constraints"
    if all_core_match and not informative_conflicts:
        return "uncertain", "all observable constraints match; qrels may be incomplete"
    if informative_conflicts:
        return "verified", "at least one explicit financial constraint conflicts"
    if "random_easy_negative" in types and candidate_doc not in gold_docs:
        return "verified", "distant candidate from another report"
    if "constraint_free_visual_negative" in types:
        return "verified", "high retrieval similarity for a query with no parsed financial constraint"
    return "uncertain", "insufficient evidence to exclude false negative"


def image_path(language: str, corpus_id: str, en_root: Path, zh_root: Path) -> Path:
    return (en_root if language == "en" else zh_root) / corpus_id


def choose_pairs(
    row: dict[str, Any],
    constraints_row: dict[str, Any],
    texts: dict[str, str],
    language: str,
    split: str,
    rng: random.Random,
    en_root: Path,
    zh_root: Path,
    max_pairs: int,
    false_negative_verification: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qrels = {cid for cid, rel in row.get("qrels", {}).items() if int(rel) > 0}
    valid_positives = [cid for cid in sorted(qrels) if image_path(language, cid, en_root, zh_root).exists()]
    if not valid_positives:
        return [], []
    positive = valid_positives[0]
    gold_docs = {page_document_id(cid) for cid in qrels}
    proposals = []
    uncertain = []
    for rank, candidate in enumerate(row.get("candidates", []), start=1):
        cid = candidate["corpus_id"]
        if cid in qrels or not image_path(language, cid, en_root, zh_root).exists():
            continue
        types, dims = classify_negative(candidate, texts.get(cid, ""), constraints_row["constraints"], gold_docs, rank)
        status, reason = verified_status(types, dims, page_document_id(cid), gold_docs, cid not in qrels)
        record = {
            "query_id": row["query_id"],
            "language": language,
            "split": split,
            "group_id": constraints_row["group_id"],
            "query": row.get("query", ""),
            "query_type": constraints_row["query_type"],
            "constraints": constraints_row["constraints"],
            "positive_page": positive,
            "positive_image": str(image_path(language, positive, en_root, zh_root)),
            "negative_page": cid,
            "negative_image": str(image_path(language, cid, en_root, zh_root)),
            "negative_rank": rank,
            "colqwen2_score": float(candidate.get("score", 0.0)),
            "negative_types": types,
            "evidence_dimensions": dims,
            "verification_status": status,
            "verification_reason": reason,
            "annotation_source": "automatic_rule_and_qrels",
        }
        if status == "verified" or not false_negative_verification:
            proposals.append(record)
        elif status == "uncertain":
            uncertain.append(record)
    priorities = [
        "single_constraint_counterfactual",
        "same_metric_wrong_year",
        "same_year_wrong_metric",
        "wrong_year",
        "wrong_metric",
        "wrong_company_entity",
        "wrong_reporting_period",
        "wrong_unit_currency",
        "wrong_statement_type",
        "wrong_accounting_scope",
        "same_document_wrong_page",
        "visually_similar_negative",
        "random_easy_negative",
    ]
    selected = []
    used_pages = set()
    for negative_type in priorities:
        options = [item for item in proposals if negative_type in item["negative_types"] and item["negative_page"] not in used_pages]
        options.sort(key=lambda item: item["negative_rank"] if negative_type != "random_easy_negative" else -item["negative_rank"])
        if options and len(selected) < max_pairs:
            choice = options[0] if negative_type != "random_easy_negative" else rng.choice(options[: min(10, len(options))])
            choice = dict(choice)
            choice["curriculum_stage"] = 0 if negative_type == "random_easy_negative" else (1 if negative_type in {"same_document_wrong_page", "visually_similar_negative"} else 2)
            selected.append(choice)
            used_pages.add(choice["negative_page"])
    return selected, uncertain


def write_samples(path: Path, pairs: list[dict[str, Any]], uncertain: list[dict[str, Any]]) -> None:
    lines = ["# FinConRank Hard Negative Samples", "", "All labels are automatically constructed. `uncertain` examples are excluded from verified training data.", ""]
    for index, item in enumerate((pairs[:80] + uncertain[:20]), start=1):
        lines.extend([
            f"## Sample {index}: `{item['query_id']}`",
            "",
            f"- Language/split: `{item['language']}` / `{item['split']}`",
            f"- Query: {item['query']}",
            f"- Positive: `{item['positive_page']}`",
            f"- Negative: `{item['negative_page']}` (ColQwen2 rank {item['negative_rank']})",
            f"- Types: `{', '.join(item['negative_types'])}`",
            f"- Verification: `{item['verification_status']}` - {item['verification_reason']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/retrieval/rerun_tab_safe_20260624_074152"))
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged"))
    parser.add_argument("--en-image-root", type=Path, default=Path("cache/official_data/corpus/en/img_origin"))
    parser.add_argument("--zh-image-root", type=Path, default=Path("cache/official_data_ch/corpus/ch/img"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-query", type=int, default=6)
    parser.add_argument("--disable-false-negative-verification", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    args.training_dir.mkdir(parents=True, exist_ok=True)
    constraints = {row["query_id"]: row for row in read_jsonl(args.run_dir / "query_constraints.jsonl")}
    rng = random.Random(args.seed)
    all_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_uncertain = []
    missing_images = []
    for language in ("en", "zh"):
        rows = read_jsonl(args.baseline_dir / f"colqwen2_{language}_predictions.jsonl")
        wanted = {candidate["corpus_id"] for row in rows for candidate in row.get("candidates", [])}
        wanted.update(cid for row in rows for cid in row.get("qrels", {}))
        texts = load_selected_text(args.index_dir / f"{language}_pages.jsonl", wanted)
        for row in rows:
            info = constraints[row["query_id"]]
            split = info["split"]
            if split == "test" and not args.include_test:
                continue
            pairs, uncertain = choose_pairs(
                row, info, texts, language, split, rng, args.en_image_root, args.zh_image_root,
                args.max_pairs_per_query, not args.disable_false_negative_verification,
            )
            all_pairs[split].extend(pairs)
            all_uncertain.extend(uncertain)
            for cid in row.get("qrels", {}):
                path = image_path(language, cid, args.en_image_root, args.zh_image_root)
                if not path.exists():
                    missing_images.append({"language": language, "query_id": row["query_id"], "corpus_id": cid, "path": str(path)})
    write_jsonl(args.training_dir / "train_pairs.jsonl", all_pairs["train"])
    write_jsonl(args.training_dir / "dev_pairs.jsonl", all_pairs["dev"])
    if args.include_test:
        write_jsonl(args.training_dir / "test_pairs.jsonl", all_pairs["test"])
    write_jsonl(args.training_dir / "uncertain_pairs.jsonl", all_uncertain)
    type_counts = Counter(kind for split in ("train", "dev") for row in all_pairs[split] for kind in row["negative_types"])
    stage_counts = Counter(row["curriculum_stage"] for split in ("train", "dev") for row in all_pairs[split])
    stats = {
        "seed": args.seed,
        "max_pairs_per_query": args.max_pairs_per_query,
        "false_negative_verification": not args.disable_false_negative_verification,
        "train_pairs": len(all_pairs["train"]),
        "dev_pairs": len(all_pairs["dev"]),
        "uncertain_excluded": len(all_uncertain),
        "negative_type_counts": dict(type_counts),
        "curriculum_stage_counts": {str(key): value for key, value in stage_counts.items()},
        "missing_gold_images": missing_images,
        "test_pairs_constructed": len(all_pairs["test"]),
        "counterfactual_query_generation": "not used: no independently verified positive page for altered queries; page-level single-constraint counterfactual negatives are used instead",
    }
    (args.training_dir / "hard_negative_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_samples(args.training_dir / "hard_negative_samples.md", all_pairs["train"] + all_pairs["dev"], all_uncertain)


if __name__ == "__main__":
    main()
