#!/usr/bin/env python3
"""Generate authorized cached A1/A2 diagnostics and blank H1 materials."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SYSTEMS = (
    "colqwen2", "monoqwen2vl", "replacement", "serial", "mono_metric_aware",
    "replacement_metric_aware", "serial_metric_aware", "full_score_fusion",
    "selected_complete_system",
)
FIELDS = (
    "entity", "canonical_metrics", "metric_terms", "year_terms", "period_terms",
    "unit_terms", "statement_type", "operation_terms", "accounting_scope",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def unique_ids(candidates: list[dict[str, Any]]) -> list[str]:
    seen = set()
    output = []
    for row in sorted(candidates, key=lambda item: int(item.get("rank", 10**9))):
        cid = str(row["corpus_id"])
        if cid not in seen:
            seen.add(cid)
            output.append(cid)
    return output


def precision_at(candidates: list[dict[str, Any]], qrels: dict[str, Any], k: int) -> float:
    relevant = {str(cid) for cid, value in qrels.items() if float(value) > 0}
    return sum(cid in relevant for cid in unique_ids(candidates)[:k]) / k


def field_present(constraints: dict[str, Any], field: str) -> bool:
    value = constraints.get(field)
    if field == "statement_type":
        return bool(value and value != "unknown")
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def run_a1_a2(strict_run: Path, revision_run: Path, evidence_index: Path) -> None:
    output_dir = revision_run / "results" / "cached_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = strict_run / "strict_nested_outer_query_predictions.jsonl"
    rows = read_jsonl(predictions_path)
    if len(rows) != 1392 or len({(row["language"], row["query_id"]) for row in rows}) != 1392:
        raise RuntimeError("strict prediction coverage is not 1,392 unique language/query pairs")
    query_metrics = []
    for row in rows:
        for system in SYSTEMS:
            base = row["systems"][system]["metrics"]
            candidates = row["systems"][system]["candidates"]
            query_metrics.append({
                "language": row["language"], "outer_fold": row["outer_fold"],
                "source_report_group": row["source_report_group"], "query_id": row["query_id"],
                "query_type": row["query_type"], "system": system,
                **{metric: float(base[metric]) for metric in ("nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10")},
                "Precision@5": precision_at(candidates, row["qrels"], 5),
                "Precision@10": precision_at(candidates, row["qrels"], 10),
                "returned_unique": len(unique_ids(candidates)),
            })
    write_csv(output_dir / "a1_query_level_metrics.csv", query_metrics)
    summary = []
    for language in ("en", "zh"):
        for system in SYSTEMS:
            subset = [row for row in query_metrics if row["language"] == language and row["system"] == system]
            summary.append({
                "language": language, "system": system, "query_count": len(subset),
                **{metric: float(np.mean([row[metric] for row in subset])) for metric in (
                    "nDCG@5", "nDCG@10", "Recall@5", "Recall@10", "MRR@10", "Precision@5", "Precision@10"
                )},
            })
    write_csv(output_dir / "a1_metrics_with_precision.csv", summary)
    definitions = """# A1 supplementary metric definitions

- Source: frozen strict report-group nested OOF query predictions; no model inference or parameter selection was performed.
- Relevance is binary: a corpus ID is relevant iff its qrels value is greater than zero.
- Candidate IDs are sorted by stored rank and deduplicated by first occurrence before scoring.
- Unjudged pages are treated as non-relevant, consistent with the existing retrieval evaluation.
- Precision@k is relevant retrieved pages in the first k unique results divided by exactly k. A query returning fewer than k pages retains denominator k; an empty query scores zero.
- nDCG, Recall, and MRR values are copied from and independently aggregated over the frozen per-query metrics.
- Precision@5/@10 are revision-driven descriptive metrics. No significance tests were selected or run for them.
- Automatic parser fields or training labels are not independent ground truth, so no constraint accuracy is reported.
"""
    (output_dir / "a1_metric_definitions.md").write_text(definitions, encoding="utf-8")

    frozen_constraints = {}
    candidate_sources = {}
    for language in ("en", "zh"):
        path = strict_run / "frozen_inputs" / f"{language}_candidates_without_qrels.jsonl"
        for row in read_jsonl(path):
            frozen_constraints[(language, row["query_id"])] = row["constraints"]
            candidate_sources[(language, row["query_id"])] = row
    diagnostics = []
    for language in ("en", "zh"):
        for outer_fold in range(5):
            subset = [row for row in rows if row["language"] == language and int(row["outer_fold"]) == outer_fold]
            diagnostic = {"language": language, "outer_fold": outer_fold, "query_count": len(subset)}
            for field in FIELDS:
                diagnostic[f"{field}_present_count"] = sum(
                    field_present(frozen_constraints[(language, row["query_id"])], field) for row in subset
                )
                diagnostic[f"{field}_coverage"] = diagnostic[f"{field}_present_count"] / len(subset)
            diagnostic["unknown_count"] = sum(
                bool(frozen_constraints[(language, row["query_id"])].get("unknown")) for row in subset
            )
            for system in ("colqwen2", "monoqwen2vl", "full_score_fusion"):
                diagnostic[f"{system}_Recall@10"] = float(np.mean([
                    row["systems"][system]["metrics"]["Recall@10"] for row in subset
                ]))
            diagnostics.append(diagnostic)
    write_csv(output_dir / "a2_parser_and_retrieval_diagnostics_by_fold.csv", diagnostics)

    parser_rows = []
    for row in rows:
        constraints = frozen_constraints[(row["language"], row["query_id"])]
        parser_rows.append({
            "language": row["language"], "outer_fold": row["outer_fold"],
            "query_id": row["query_id"], "source_report_group": row["source_report_group"],
            "query_type": row["query_type"], "parser_confidence": constraints.get("confidence"),
            "unknown": constraints.get("unknown"),
            **{f"{field}_present": field_present(constraints, field) for field in FIELDS},
        })
    write_csv(output_dir / "a2_query_parser_field_audit.csv", parser_rows)

    evidence_rows = []
    for language in ("en", "zh"):
        pages = {row["corpus_id"]: row for row in read_jsonl(evidence_index / f"{language}_pages.jsonl")}
        for outer_fold in range(5):
            subset = [row for row in rows if row["language"] == language and int(row["outer_fold"]) == outer_fold]
            candidate_ids = [
                candidate["corpus_id"]
                for row in subset
                for candidate in candidate_sources[(language, row["query_id"])]["colqwen2_candidates"][:20]
            ]
            gold_ids = [cid for row in subset for cid, rel in row["qrels"].items() if float(rel) > 0]
            for unit, ids in (("colqwen2_top20_candidate_pair", candidate_ids), ("qrels_positive_page_reference", gold_ids)):
                observed = [pages.get(cid) for cid in ids]
                evidence_rows.append({
                    "language": language, "outer_fold": outer_fold, "unit": unit, "references": len(ids),
                    "page_index_match_count": sum(item is not None for item in observed),
                    "has_text_count": sum(bool(item and item.get("has_text")) for item in observed),
                    "ocr_text_count": sum(bool(item and int(item.get("ocr_word_count") or 0) > 0) for item in observed),
                    "pdf_native_text_count": sum(bool(item and item.get("pdf_has_text")) for item in observed),
                    "missing_page_index_count": sum(item is None for item in observed),
                })
    write_csv(output_dir / "a2_ocr_pdf_availability_by_fold.csv", evidence_rows)

    selected_examples = []
    for language in ("en", "zh"):
        pool = [row for row in rows if row["language"] == language]
        ranked = sorted(
            pool,
            key=lambda row: (
                -sum(field_present(frozen_constraints[(language, row["query_id"])], field) for field in FIELDS),
                row["query_id"],
            ),
        )
        chosen_types = set()
        for row in ranked:
            if row["query_type"] in chosen_types and len(selected_examples) % 20 < 12:
                continue
            selected_examples.append(row)
            chosen_types.add(row["query_type"])
            if sum(item["language"] == language for item in selected_examples) == 20:
                break
    lines = ["# A2 bilingual query-to-constraint examples", "", "These are deterministic parser outputs from frozen strict inputs, not independently verified labels.", ""]
    for row in selected_examples:
        constraints = frozen_constraints[(row["language"], row["query_id"])]
        lines.extend([
            f"## {row['language'].upper()} `{row['query_id']}`",
            f"- Query type: `{row['query_type']}`; report group: `{row['source_report_group']}`; outer fold: `{row['outer_fold']}`",
            f"- Query: {row['query']}",
            f"- Parsed constraints: `{json.dumps(constraints, ensure_ascii=False, sort_keys=True)}`",
            "- Interpretation boundary: missing fields are unobserved parser outputs, not proven semantic conflicts.", "",
        ])
    (output_dir / "a2_query_constraint_examples.md").write_text("\n".join(lines), encoding="utf-8")

    improvements = []
    regressions = []
    for row in rows:
        mono = row["systems"]["monoqwen2vl"]["metrics"]["nDCG@10"]
        full = row["systems"]["full_score_fusion"]["metrics"]["nDCG@10"]
        target = improvements if full > mono else regressions if full < mono else None
        if target is not None:
            target.append((abs(full - mono), row, full - mono))
    case_lines = ["# A2 post-hoc bilingual error examples", "", "Cases are selected descriptively from frozen predictions and were not used to change rules or tune parameters.", ""]
    for heading, values in (("Largest improvements", improvements), ("Largest regressions", regressions)):
        case_lines.extend([f"## {heading}", ""])
        for _, row, delta in sorted(values, reverse=True, key=lambda item: item[0])[:10]:
            constraints = frozen_constraints[(row["language"], row["query_id"])]
            case_lines.extend([
                f"- `{row['language']}` `{row['query_id']}` ({row['query_type']}, group `{row['source_report_group']}`): delta nDCG@10={delta:.6f}",
                f"  Query: {row['query']}",
                f"  Parser output: `{json.dumps(constraints, ensure_ascii=False, sort_keys=True)}`",
            ])
        case_lines.append("")
    (output_dir / "a2_error_examples.md").write_text("\n".join(case_lines), encoding="utf-8")
    audit = {
        "source_predictions": str(predictions_path.resolve()), "source_sha256": sha256(predictions_path),
        "query_count": len(rows), "languages": dict(Counter(row["language"] for row in rows)),
        "query_metric_rows": len(query_metrics), "precision_is_descriptive": True,
        "constraint_accuracy_computed": False, "model_inference_run": False, "parameters_selected": False,
        "evidence_index": str(evidence_index.resolve()),
        "status": "PASS",
    }
    (output_dir / "a1_a2_completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def run_h1(revision_run: Path, blinded_path: Path) -> None:
    output_dir = revision_run / "results" / "human_audit_h1"
    output_dir.mkdir(parents=True, exist_ok=True)
    with blinded_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200 or len({row["sample_id"] for row in rows}) != 200:
        raise RuntimeError("H1 source does not contain exactly 200 unique samples")
    label_fields = (
        "positive_page_can_answer", "negative_page_can_answer", "negative_is_valid",
        "conflict_type", "query_is_natural", "notes", "annotator_id", "confidence_1_to_5",
    )
    for row in rows:
        for field in label_fields:
            row[field] = ""
    output = output_dir / "second_annotator_blinded_blank.csv"
    write_csv(output, rows)
    instructions = """# Independent second-annotator form (blank; awaiting real human input)

This form preserves the fixed 200 sample IDs and contains no first-annotator labels, automatic conflict labels, model scores, rankings, qrels flags, or Generic/Financial source indicator. A real second annotator must fill every judgment independently. No LLM, rule system, or Codex output may be entered as human annotation.

Fields follow the existing first-round protocol: whether the positive and candidate pages independently answer the query, whether the candidate is a valid negative, zero or more conflict types, query naturalness, notes, annotator ID, and confidence. `uncertain` is allowed. Images are primary evidence and OCR is auxiliary.

Until a real second annotator returns this form, inter-annotator agreement is **not available** and H1 remains `WAITING_HUMAN_INPUT`.
"""
    (output_dir / "second_annotator_instructions.md").write_text(instructions, encoding="utf-8")
    audit = {
        "source": str(blinded_path.resolve()), "source_sha256": sha256(blinded_path),
        "output": str(output.resolve()), "output_sha256": sha256(output),
        "rows": len(rows), "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "all_human_fields_blank": all(not row[field] for row in rows for field in label_fields),
        "status": "WAITING_HUMAN_INPUT", "human_labels_fabricated": False,
    }
    (output_dir / "h1_status.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--revision-run", type=Path, required=True)
    parser.add_argument("--human-blinded", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    args = parser.parse_args()
    run_a1_a2(args.strict_run.resolve(), args.revision_run.resolve(), args.evidence_index.resolve())
    run_h1(args.revision_run.resolve(), args.human_blinded.resolve())


if __name__ == "__main__":
    main()
