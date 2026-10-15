#!/usr/bin/env python3
"""Merge Tesseract OCR indexes with PDF-native indexes.

OCR page text and OCR line regions take precedence where available. PDF-native
text/blocks are retained as fallback for pages not yet covered by OCR.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def merge_language(language: str, pdf_dir: Path, ocr_dir: Path, out_dir: Path) -> dict[str, Any]:
    pdf_pages = {row["corpus_id"]: row for row in read_jsonl(pdf_dir / f"{language}_pages.jsonl")}
    ocr_pages = {row["corpus_id"]: row for row in read_jsonl(ocr_dir / f"{language}_ocr_pages.jsonl")}
    merged_pages = []
    for cid, pdf_row in pdf_pages.items():
        ocr_row = ocr_pages.get(cid)
        if ocr_row and ocr_row.get("has_ocr_text"):
            merged_pages.append(
                {
                    "corpus_id": cid,
                    "title": ocr_row.get("title") or pdf_row.get("title"),
                    "text": ocr_row.get("text") or pdf_row.get("text"),
                    "has_text": True,
                    "text_source": "tesseract_page_image",
                    "image_path": ocr_row.get("image_path"),
                    "ocr_word_count": ocr_row.get("ocr_word_count", 0),
                    "pdf_has_text": pdf_row.get("has_pdf_text", False),
                }
            )
        else:
            merged_pages.append(
                {
                    "corpus_id": cid,
                    "title": pdf_row.get("title"),
                    "text": pdf_row.get("text"),
                    "has_text": bool(pdf_row.get("text")),
                    "text_source": "pdf_native_fallback",
                    "image_path": None,
                    "ocr_word_count": 0,
                    "pdf_has_text": pdf_row.get("has_pdf_text", False),
                }
            )

    ocr_blocks = read_jsonl(ocr_dir / f"{language}_ocr_blocks.jsonl")
    ocr_block_pages = {row["corpus_id"] for row in ocr_blocks}
    pdf_blocks = [row for row in read_jsonl(pdf_dir / f"{language}_blocks.jsonl") if row["corpus_id"] not in ocr_block_pages]
    merged_blocks = ocr_blocks + pdf_blocks

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{language}_pages.jsonl", merged_pages)
    write_jsonl(out_dir / f"{language}_blocks.jsonl", merged_blocks)
    stats = {
        "language": language,
        "pdf_pages": len(pdf_pages),
        "ocr_pages": len(ocr_pages),
        "ocr_pages_with_text": sum(1 for row in ocr_pages.values() if row.get("has_ocr_text")),
        "merged_pages": len(merged_pages),
        "merged_pages_using_ocr": sum(1 for row in merged_pages if row["text_source"] == "tesseract_page_image"),
        "pdf_blocks_retained": len(pdf_blocks),
        "ocr_blocks": len(ocr_blocks),
        "merged_blocks": len(merged_blocks),
    }
    (out_dir / f"{language}_merge_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-index-dir", type=Path, required=True)
    parser.add_argument("--ocr-index-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    stats = {
        "en": merge_language("en", args.pdf_index_dir, args.ocr_index_dir, args.out_dir),
        "zh": merge_language("zh", args.pdf_index_dir, args.ocr_index_dir, args.out_dir),
    }
    (args.out_dir / "index_audit.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
