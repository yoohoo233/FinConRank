from __future__ import annotations

from finconrank.release_paths import release_path

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(os.environ.get("FINCONRANK_BASELINE_DIR") or PROJECT / "artifacts/retrieval/rerun_tab_safe_20260624_074152")
SOURCE_RUN = Path(os.environ.get("FINCONRANK_SOURCE_RUN") or PROJECT / "artifacts/retrieval/finconrank_20260807_184746")
SOURCE_OOF = Path(os.environ.get("FINCONRANK_SOURCE_OOF") or PROJECT / "artifacts/training/finconrank_20260807_184746_oof")
INDEX_DIR = Path(os.environ.get("FINCONRANK_INDEX_DIR") or PROJECT / "artifacts/indexes/real_router_multigranularity_20260624_131957_candidate_ocr_pdf_merged")
ROUTER_DIR = Path(os.environ.get("FINCONRANK_ROUTER_DIR") or PROJECT / "artifacts/retrieval/real_router_multigranularity_20260624_131957/candidate_ocr_pdf_merged_rerun")
CONSTRAINTS = Path(os.environ.get("FINCONRANK_CONSTRAINTS") or SOURCE_RUN / "query_constraints.jsonl")
BASE_MODEL = Path(release_path('FINCONRANK_BASE_MODEL', 'models/Qwen2-VL-2B-Instruct'))
INITIAL_ADAPTER = Path(release_path('FINCONRANK_INITIAL_ADAPTER', 'models/MonoQwen2-VL-v0.1'))
IMAGE_ROOTS = {
    "en": Path(os.environ.get("FINCONRANK_EN_IMAGES") or PROJECT / "cache/official_data/corpus/en/img_origin"),
    "zh": Path(os.environ.get("FINCONRANK_ZH_IMAGES") or PROJECT / "cache/official_data_ch/corpus/ch/img"),
}
LANGUAGES = ("en", "zh")
OUTER_FOLDS = 5
INNER_FOLDS = 4
CANDIDATE_DEPTH = 20
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_paths() -> dict[str, Path]:
    return {
        "constraints": CONSTRAINTS,
        "source_split_manifest": SOURCE_RUN / "split_manifest.json",
        "source_outer_manifest": SOURCE_OOF / "oof_manifest.json",
        "colqwen2_en": BASELINE_DIR / "colqwen2_en_predictions.jsonl",
        "colqwen2_zh": BASELINE_DIR / "colqwen2_zh_predictions.jsonl",
        "monoqwen2vl_en": SOURCE_RUN / "mono_top50/en_monoqwen2vl_top50_predictions.jsonl",
        "monoqwen2vl_zh": SOURCE_RUN / "mono_top50/zh_monoqwen2vl_top50_predictions.jsonl",
        "ocr_pages_en": INDEX_DIR / "en_pages.jsonl",
        "ocr_pages_zh": INDEX_DIR / "zh_pages.jsonl",
        "ocr_blocks_en": INDEX_DIR / "en_blocks.jsonl",
        "ocr_blocks_zh": INDEX_DIR / "zh_blocks.jsonl",
        "router_scores_en": ROUTER_DIR / "en_query_router_multi_granularity_mono_predictions.jsonl",
        "router_scores_zh": ROUTER_DIR / "zh_query_router_multi_granularity_mono_predictions.jsonl",
    }


def candidate_cache_path(run_dir: Path, language: str) -> Path:
    return run_dir / f"frozen_inputs/{language}_candidates_without_qrels.jsonl"


def load_candidate_cache(run_dir: Path, language: str) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in read_jsonl(candidate_cache_path(run_dir, language))}


def qrels_path(run_dir: Path, language: str, outer_fold: int, split: str) -> Path:
    return run_dir / f"sealed_qrels/{language}_outer_{outer_fold}_{split}_qrels.json"


def task_id(kind: str, language: str, outer_fold: int, inner_fold: int | None = None) -> str:
    if kind == "inner":
        if inner_fold is None:
            raise ValueError("inner fold is required")
        return f"inner_{language}_outer{outer_fold}_inner{inner_fold}"
    return f"outer_{language}_fold{outer_fold}"


def task_dir(run_dir: Path, identifier: str) -> Path:
    return run_dir / "tasks" / identifier


def union_candidates(row: dict[str, Any], depth: int = CANDIDATE_DEPTH) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source_name, rank_name, values in (
        ("colqwen2_top20", "colqwen2_rank", row["colqwen2_candidates"][:depth]),
        ("monoqwen2vl_top20", "monoqwen2vl_rank", row["monoqwen2vl_candidates"][:depth]),
    ):
        for rank, source in enumerate(values, 1):
            cid = source["corpus_id"]
            if cid not in merged:
                merged[cid] = {
                    "candidate_id": cid,
                    "candidate_sources": [],
                    "colqwen2_rank": None,
                    "monoqwen2vl_rank": None,
                    "colqwen2_score": None,
                    "monoqwen2vl_score": None,
                }
                order.append(cid)
            item = merged[cid]
            item["candidate_sources"].append(source_name)
            item[rank_name] = rank
            if source_name == "colqwen2_top20":
                item["colqwen2_score"] = source.get("score")
            else:
                item["monoqwen2vl_score"] = source.get("rerank_score")
                if item["colqwen2_score"] is None:
                    item["colqwen2_score"] = source.get("score")
    return [merged[cid] for cid in order]


def descendants_of(pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            right = stat.rfind(")")
            ppid = int(stat[right + 2 :].split()[1])
            children.setdefault(ppid, []).append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    found = {pid}
    queue = [pid]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found
