#!/usr/bin/env python3
"""Build real page-text and PDF text-block indexes for FinRAGBench-V.

The script reads original PDF tarballs and extracted page image ids, writes all
derived artifacts under <repository-root>, and never modifies <read-only-data-root>.
"""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

from finconrank.release_paths import release_path

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def corpus_ids(image_root: Path) -> set[str]:
    return {path.name for path in image_root.rglob("*.png")}


def all_doc_titles(ids: set[str]) -> dict[str, str]:
    titles = {}
    for cid in ids:
        if "_" in cid:
            titles[cid] = cid.rsplit("_", 1)[0]
        else:
            titles[cid] = cid.removesuffix(".png")
    return titles


def extract_page_blocks(page: Any) -> tuple[str, list[dict[str, Any]]]:
    fragments: list[dict[str, Any]] = []

    def visitor_text(text, cm, tm, font_dict, font_size):
        clean = " ".join(str(text).split())
        if not clean:
            return
        x = float(tm[4]) if len(tm) > 4 else 0.0
        y = float(tm[5]) if len(tm) > 5 else 0.0
        fragments.append({"text": clean, "x": x, "y": y, "font_size": float(font_size or 0.0)})

    try:
        text = page.extract_text(visitor_text=visitor_text) or ""
    except Exception:
        text = ""
        fragments = []
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

    # Group fragments by coarse y-coordinate bands into real PDF text regions.
    grouped: dict[int, list[dict[str, Any]]] = {}
    for frag in fragments:
        band = int(round(frag["y"] / 24.0))
        grouped.setdefault(band, []).append(frag)
    blocks = []
    for index, (_, items) in enumerate(sorted(grouped.items(), key=lambda kv: -kv[0])):
        items.sort(key=lambda item: item["x"])
        block_text = " ".join(item["text"] for item in items).strip()
        if len(block_text) < 3:
            continue
        xs = [item["x"] for item in items]
        ys = [item["y"] for item in items]
        blocks.append(
            {
                "block_index": index,
                "text": block_text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "source": "pdf_text_fragment_region",
            }
        )
    if not text and blocks:
        text = "\n".join(block["text"] for block in blocks)
    return text, blocks


def build_language(language: str, pdf_tar: Path, image_root: Path, out_dir: Path) -> dict[str, Any]:
    ids = corpus_ids(image_root)
    titles = all_doc_titles(ids)
    pages: dict[str, dict[str, Any]] = {
        cid: {"corpus_id": cid, "title": title, "text": title, "has_pdf_text": False, "source_pdf": None}
        for cid, title in titles.items()
    }
    blocks: list[dict[str, Any]] = []
    pdf_count = 0
    pdf_errors = []
    matched_pages = 0
    with tarfile.open(pdf_tar, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            pdf_count += 1
            stem = Path(member.name).stem
            try:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                data = handle.read()
                reader = PdfReader(io.BytesIO(data))
            except Exception as exc:
                pdf_errors.append({"pdf": member.name, "error": type(exc).__name__, "message": str(exc)[:200]})
                continue
            try:
                page_iter = list(reader.pages)
            except Exception as exc:
                pdf_errors.append({"pdf": member.name, "error": type(exc).__name__, "message": str(exc)[:200]})
                continue
            for page_index, page in enumerate(page_iter, start=1):
                corpus_id = f"{stem}_{page_index}.png"
                if corpus_id not in ids:
                    continue
                matched_pages += 1
                try:
                    text, page_blocks = extract_page_blocks(page)
                except Exception as exc:
                    pdf_errors.append({"pdf": member.name, "page": page_index, "error": type(exc).__name__, "message": str(exc)[:200]})
                    continue
                page_text = " ".join([stem, text]).strip()
                pages[corpus_id] = {
                    "corpus_id": corpus_id,
                    "title": stem,
                    "text": page_text or stem,
                    "has_pdf_text": bool(text.strip()),
                    "source_pdf": member.name,
                }
                for block in page_blocks:
                    if block["text"].strip():
                        blocks.append(
                            {
                                "block_id": f"{corpus_id}#b{block['block_index']}",
                                "corpus_id": corpus_id,
                                "title": stem,
                                **block,
                            }
                        )

    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = out_dir / f"{language}_pages.jsonl"
    block_path = out_dir / f"{language}_blocks.jsonl"
    page_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages.values()), encoding="utf-8")
    block_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blocks), encoding="utf-8")
    stats = {
        "language": language,
        "pdf_tar": str(pdf_tar),
        "image_root": str(image_root),
        "corpus_pages": len(ids),
        "pdfs_scanned": pdf_count,
        "matched_pdf_pages": matched_pages,
        "pages_with_pdf_text": sum(1 for row in pages.values() if row["has_pdf_text"]),
        "blocks": len(blocks),
        "page_index": str(page_path),
        "block_index": str(block_path),
        "pdf_errors_sample": pdf_errors[:50],
        "pdf_error_count": len(pdf_errors),
    }
    (out_dir / f"{language}_index_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(release_path('FINRAGBENCH_DATA_ROOT', 'data/FinRAGBench-V', '')))
    parser.add_argument("--en-image-root", type=Path, default=Path("cache/official_data/corpus/en/img_origin"))
    parser.add_argument("--zh-image-root", type=Path, default=Path("cache/official_data_ch/corpus/ch/img"))
    args = parser.parse_args()
    stats = {
        "en": build_language("en", args.data_root / "pdfs_for_QA/pdf_en.tar.gz", args.en_image_root, args.out_dir),
        "zh": build_language("zh", args.data_root / "pdfs_for_QA/pdf_ch.tar.gz", args.zh_image_root, args.out_dir),
    }
    (args.out_dir / "index_audit.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
