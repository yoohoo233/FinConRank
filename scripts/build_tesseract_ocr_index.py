#!/usr/bin/env python3
"""Build page-image OCR text and region indexes with Tesseract TSV output."""

from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def image_paths(image_root: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(image_root.rglob("*.png"))
    return paths[:limit] if limit else paths


def manifest_image_paths(image_root: Path, manifest: Path, limit: int | None = None) -> list[Path]:
    paths = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            cid = line.strip()
            if not cid:
                continue
            path = image_root / cid
            if path.exists():
                paths.append(path)
    paths = sorted(paths)
    return paths[:limit] if limit else paths


def run_tesseract_tsv(path: Path, lang: str, timeout: int, omp_thread_limit: int | None) -> tuple[str, list[dict[str, Any]], str | None]:
    cmd = ["tesseract", str(path), "stdout", "-l", lang, "--psm", "6", "tsv"]
    env = os.environ.copy()
    if omp_thread_limit:
        env["OMP_THREAD_LIMIT"] = str(omp_thread_limit)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired:
        return "", [], "TimeoutExpired"
    if proc.returncode != 0 and not proc.stdout:
        return "", [], (proc.stderr or f"returncode={proc.returncode}")[:300]
    rows = []
    reader = csv.DictReader(proc.stdout.splitlines(), delimiter="\t")
    try:
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(row.get("conf") or -1)
                left = int(float(row.get("left") or 0))
                top = int(float(row.get("top") or 0))
                width = int(float(row.get("width") or 0))
                height = int(float(row.get("height") or 0))
                level = int(float(row.get("level") or 0))
            except ValueError:
                continue
            if conf < 0:
                continue
            rows.append(
                {
                    "level": level,
                    "text": text,
                    "conf": conf,
                    "bbox": [left, top, left + width, top + height],
                    "block_num": row.get("block_num"),
                    "par_num": row.get("par_num"),
                    "line_num": row.get("line_num"),
                    "word_num": row.get("word_num"),
                    "source": "tesseract_tsv",
                }
            )
    except csv.Error as exc:
        return "", [], f"csv.Error: {exc}"
    text = " ".join(row["text"] for row in rows)
    return text, rows, None


def line_blocks(corpus_id: str, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str | None, str | None, str | None], list[dict[str, Any]]] = {}
    for word in words:
        key = (word.get("block_num"), word.get("par_num"), word.get("line_num"))
        grouped.setdefault(key, []).append(word)
    blocks = []
    for idx, (_, items) in enumerate(grouped.items()):
        items.sort(key=lambda row: (row["bbox"][1], row["bbox"][0]))
        text = " ".join(item["text"] for item in items).strip()
        if len(text) < 2:
            continue
        xs0 = [item["bbox"][0] for item in items]
        ys0 = [item["bbox"][1] for item in items]
        xs1 = [item["bbox"][2] for item in items]
        ys1 = [item["bbox"][3] for item in items]
        conf = sum(item["conf"] for item in items) / len(items)
        blocks.append(
            {
                "block_id": f"{corpus_id}#ocr_l{idx}",
                "corpus_id": corpus_id,
                "block_index": idx,
                "text": text,
                "bbox": [min(xs0), min(ys0), max(xs1), max(ys1)],
                "confidence": conf,
                "source": "tesseract_line_region",
            }
        )
    return blocks


def process_one(path: Path, image_root: Path, lang: str, timeout: int, omp_thread_limit: int | None) -> dict[str, Any]:
    corpus_id = path.name
    title = path.name.rsplit("_", 1)[0] if "_" in path.name else path.stem
    text, words, error = run_tesseract_tsv(path, lang, timeout, omp_thread_limit)
    blocks = line_blocks(corpus_id, words)
    return {
        "page": {
            "corpus_id": corpus_id,
            "title": title,
            "image_path": str(path.relative_to(image_root)),
            "text": " ".join([title, text]).strip(),
            "ocr_text": text,
            "has_ocr_text": bool(text.strip()),
            "ocr_word_count": len(words),
            "ocr_block_count": len(blocks),
            "ocr_error": error,
            "source": "tesseract_page_image",
        },
        "blocks": blocks,
    }


def load_done(page_path: Path) -> set[str]:
    if not page_path.exists():
        return set()
    done = set()
    with page_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["corpus_id"])
            except Exception:
                continue
    return done


def main() -> None:
    raise_csv_field_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "zh"], required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", type=Path, help="Optional newline corpus_id list to OCR instead of the full image root.")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--omp-thread-limit", type=int, default=1, help="Set OMP_THREAD_LIMIT for each Tesseract subprocess.")
    args = parser.parse_args()

    tess_lang = "eng" if args.language == "en" else "chi_sim+eng"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    page_path = args.out_dir / f"{args.language}_ocr_pages.jsonl"
    block_path = args.out_dir / f"{args.language}_ocr_blocks.jsonl"
    stats_path = args.out_dir / f"{args.language}_ocr_stats.json"
    done = load_done(page_path) if args.resume else set()
    source_paths = manifest_image_paths(args.image_root, args.manifest, args.limit) if args.manifest else image_paths(args.image_root, args.limit)
    paths = [path for path in source_paths if path.name not in done]

    mode = "a" if args.resume else "w"
    processed = 0
    errors = 0
    pages_with_text = 0
    blocks = 0
    with page_path.open(mode, encoding="utf-8") as page_file, block_path.open(mode, encoding="utf-8") as block_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, path, args.image_root, tess_lang, args.timeout, args.omp_thread_limit) for path in paths]
            for future in as_completed(futures):
                row = future.result()
                page_file.write(json.dumps(row["page"], ensure_ascii=False) + "\n")
                for block in row["blocks"]:
                    block_file.write(json.dumps(block, ensure_ascii=False) + "\n")
                processed += 1
                errors += bool(row["page"].get("ocr_error"))
                pages_with_text += bool(row["page"].get("has_ocr_text"))
                blocks += len(row["blocks"])
                if processed % 100 == 0:
                    print(f"{args.language}: OCR processed {processed}/{len(paths)} new pages", flush=True)

    total_done = len(load_done(page_path))
    stats = {
        "language": args.language,
        "image_root": str(args.image_root),
        "tesseract_lang": tess_lang,
        "workers": args.workers,
        "omp_thread_limit": args.omp_thread_limit,
        "limit": args.limit,
        "manifest": str(args.manifest) if args.manifest else None,
        "manifest_or_root_pages": len(source_paths),
        "new_pages_processed": processed,
        "total_pages_in_output": total_done,
        "new_pages_with_ocr_text": pages_with_text,
        "new_ocr_errors": errors,
        "new_blocks": blocks,
        "page_index": str(page_path),
        "block_index": str(block_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
