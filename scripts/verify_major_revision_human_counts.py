#!/usr/bin/env python3
"""Verify first-round human counts without filling or modifying labels."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import csv
from collections import Counter
from pathlib import Path
from finconrank.strict_nested import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["negative_is_valid"] for row in rows)
    positives = Counter(row["positive_page_can_answer"] for row in rows)
    language = {}
    for lang in sorted({row["language"] for row in rows}):
        subset = [row for row in rows if row["language"] == lang]
        language[lang] = {"denominator": len(subset), "validity": dict(Counter(row["negative_is_valid"] for row in subset))}
    result = {"source": str(args.source.resolve()), "sha256": sha256_file(args.source), "total": len(rows),
              "unique_sample_ids": len({row["sample_id"] for row in rows}), "negative_validity": dict(counts),
              "positive_independent_answerability": dict(positives), "by_language": language,
              "completed": sum(row["annotation_status"] == "completed" for row in rows),
              "boundary": "Single-annotator stratified mixed-source quality sample, not training accuracy or inter-annotator agreement. Positive no=9 and uncertain=1 are distinct. Conflict-label precision/recall measures label-set overlap, not negative validity.",
              "expected_191_8_1_pass": counts == Counter({"yes": 191, "no": 8, "uncertain": 1})}
    atomic_json(args.output, result)
    if not result["expected_191_8_1_pass"]:
        raise RuntimeError("first-round counts differ")


if __name__ == "__main__":
    main()
