#!/usr/bin/env python3
"""Independently recompute standard metrics from exported query rankings."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[1] / "src"))
import argparse
import json
import math
from pathlib import Path
from finconrank.strict_nested import atomic_json, sha256_file


def metrics(items, qrels):
    ranked = sorted(items, key=lambda x: x["rank"])
    seen, ids = set(), []
    for item in ranked:
        cid = item.get("corpus_id", item.get("candidate_id"))
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    rels = [float(qrels.get(cid, 0)) for cid in ids]
    positive = sum(float(x) > 0 for x in qrels.values())
    ideal = sorted((float(x) for x in qrels.values()), reverse=True)
    result = {}
    for k in (5, 10):
        dcg = sum(v / math.log2(i + 2) for i, v in enumerate(rels[:k]))
        idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:k]))
        result[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
        result[f"Recall@{k}"] = sum(v > 0 for v in rels[:k]) / positive if positive else 0.0
    result["MRR@10"] = next((1 / (i + 1) for i, v in enumerate(rels[:10]) if v > 0), 0.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    maximum, count, mismatch = 0.0, 0, []
    for line in args.predictions.open():
        row = json.loads(line)
        for system, values in row["systems"].items():
            actual = metrics(values["candidates"], row["qrels"])
            for name, value in actual.items():
                difference = abs(value - values["metrics"][name])
                maximum = max(maximum, difference)
                count += 1
                if difference > 1e-12:
                    mismatch.append([row["query_id"], system, name, difference])
    atomic_json(args.output, {"source_sha256": sha256_file(args.predictions), "checked_metric_values": count,
                            "maximum_absolute_error": maximum, "mismatches": mismatch, "passed": not mismatch})
    if mismatch:
        raise RuntimeError(f"{len(mismatch)} metric mismatches")


if __name__ == "__main__":
    main()
