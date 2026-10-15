#!/usr/bin/env python3
"""Prepare official-style FinRAGBench-V retrieval inputs under <repository-root>.

Follows the official README workflow: extract corpus part files, convert images
and queries to parquet, and keep qrels as TSV. Does not use the prepared/
directory from the source dataset.
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
import os
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


DATA_ROOT = Path(release_path('FINRAGBENCH_DATA_ROOT', 'data/FinRAGBench-V', ''))
OUT_ROOT = Path(release_path('FINCONRANK_WORKSPACE', '.', 'cache/official_data'))


def extract_corpus(language: str, out_root: Path, force: bool = False) -> Path:
    target = out_root / "corpus" / language
    marker = target / ".extract_complete"
    if marker.exists() and not force:
        return target
    target.mkdir(parents=True, exist_ok=True)
    part_glob = str(DATA_ROOT / "corpus" / language / "part_*")
    cmd = f"cat {part_glob} | tar -xz -C {target}"
    subprocess.run(["bash", "-lc", cmd], check=True)
    marker.write_text("ok\n", encoding="utf-8")
    return target


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() == ".png")


def images_to_parquet(input_dir: Path, output_dir: Path, shard_size: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = image_paths(input_dir)
    shards = []
    rows = []
    shard_index = 0
    for path in images:
        try:
            img_bytes = path.read_bytes()
            Image.open(io.BytesIO(img_bytes)).verify()
            rows.append({"corpus-id": path.name, "image": img_bytes})
        except Exception:
            continue
        if len(rows) >= shard_size:
            out = output_dir / f"corpus_{shard_index:05d}.parquet"
            pd.DataFrame(rows).to_parquet(out, index=False, engine="pyarrow")
            shards.append({"path": str(out), "rows": len(rows), "bytes": out.stat().st_size})
            rows = []
            shard_index += 1
    if rows:
        out = output_dir / f"corpus_{shard_index:05d}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False, engine="pyarrow")
        shards.append({"path": str(out), "rows": len(rows), "bytes": out.stat().st_size})
    return shards


def json_to_parquet(json_file: Path, parquet_file: Path) -> int:
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(json_file)
    df.to_parquet(parquet_file, index=False, engine="pyarrow")
    return len(df)


def copy_qrels(language: str, out_root: Path) -> Path:
    src = DATA_ROOT / "qrels" / f"qrels_{language}.tsv"
    dst = out_root / "dataset" / language / "qrels" / f"qrels_{language}.tsv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    extracted = args.out_root / "corpus" / args.language
    if not args.skip_extract:
        extracted = extract_corpus(args.language, args.out_root)
    corpus_parquet = args.out_root / "corpus_parquet" / args.language
    shards = images_to_parquet(extracted, corpus_parquet, args.shard_size)
    query_rows = json_to_parquet(
        DATA_ROOT / "queries" / f"queries_{args.language}.json",
        args.out_root / "queries" / f"queries_{args.language}.parquet",
    )
    qrels_path = copy_qrels(args.language, args.out_root)
    manifest = {
        "official_method": "merge part_*.tar.gz, extract, convert corpus images and queries to parquet",
        "prepared_used": False,
        "source_data_root": str(DATA_ROOT),
        "output_root": str(args.out_root),
        "language": args.language,
        "extracted_corpus": str(extracted),
        "corpus_parquet": str(corpus_parquet),
        "corpus_rows": sum(item["rows"] for item in shards),
        "query_rows": query_rows,
        "qrels_path": str(qrels_path),
        "shards": shards,
    }
    manifest_path = args.out_root / f"manifest_{args.language}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
