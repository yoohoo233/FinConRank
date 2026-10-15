#!/usr/bin/env python3
from __future__ import annotations

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from finconrank.common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--all-pairs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    qid_group = {qid: item["group_id"] for qid, item in manifest["queries"].items()}
    groups = sorted(set(qid_group.values()))
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    group_qids: dict[str, list[str]] = defaultdict(list)
    for qid, group in qid_group.items():
        group_qids[group].append(qid)
    folds: list[list[str]] = [[] for _ in range(args.folds)]
    for group in sorted(groups, key=lambda key: len(group_qids[key]), reverse=True):
        target = min(range(args.folds), key=lambda idx: sum(len(group_qids[item]) for item in folds[idx]))
        folds[target].append(group)
    all_pairs = []
    for name in ("train_pairs.jsonl", "dev_pairs.jsonl", "test_pairs.jsonl"):
        all_pairs.extend(read_jsonl(args.all_pairs_dir / name))
    fold_manifest = {"seed": args.seed, "fold_count": args.folds, "folds": []}
    for fold_id, test_groups in enumerate(folds):
        test_set = set(test_groups)
        remaining = [group for group in groups if group not in test_set]
        random.Random(args.seed + fold_id + 100).shuffle(remaining)
        dev_count = max(1, round(0.15 * len(remaining)))
        dev_set = set(remaining[:dev_count])
        train_set = set(remaining[dev_count:])
        fold_dir = args.output_dir / f"fold_{fold_id}"
        train = [row for row in all_pairs if qid_group[row["query_id"]] in train_set]
        dev = [row for row in all_pairs if qid_group[row["query_id"]] in dev_set]
        write_jsonl(fold_dir / "train_pairs.jsonl", train)
        write_jsonl(fold_dir / "dev_pairs.jsonl", dev)
        test_qids = sorted(qid for qid, group in qid_group.items() if group in test_set)
        (fold_dir / "test_qids.json").write_text(json.dumps(test_qids, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_manifest["folds"].append({
            "fold": fold_id,
            "train_groups": len(train_set),
            "dev_groups": len(dev_set),
            "test_groups": len(test_set),
            "train_pairs": len(train),
            "dev_pairs": len(dev),
            "test_queries": len(test_qids),
            "test_group_ids": sorted(test_set),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof_manifest.json").write_text(json.dumps(fold_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
