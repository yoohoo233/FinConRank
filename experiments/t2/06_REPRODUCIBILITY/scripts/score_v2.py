"""Score only locked validation queries with a newly trained fixed checkpoint."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))

from finconrank.release_paths import release_path
import argparse
import math
import os
import time
from pathlib import Path
from t2_common import RUN, read, rows, atomic, sha, verify_protocol, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    from run_finconrank_strict_nested_task import load_model, score_batch
    import torch
    records = rows(args.candidates)
    processor, model, tokens = load_model(args.adapter)
    checkpoint_hash = sha(args.adapter / "adapter_model.safetensors")
    started = time.perf_counter()
    output = []
    count = 0
    for index, row in enumerate(records, 1):
        language = row["language"]
        root = Path(release_path('FINCONRANK_WORKSPACE', '.', 'cache/official_data/corpus/en/img_origin') if language == "en" else release_path('FINCONRANK_WORKSPACE', '.', 'cache/official_data_ch/corpus/ch/img'))
        items = row["candidates"]
        scores = []
        for offset in range(0, len(items), 2):
            paths = [root / item["corpus_id"] for item in items[offset:offset + 2]]
            if any(not path.is_file() for path in paths):
                raise RuntimeError("candidate image missing; cannot skip")
            scores.extend(score_batch(processor, model, tokens, row["query"], row["constraints"], paths))
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("non-finite score")
        ranked = [{**item, "finconrank_score": score} for item, score in zip(items, scores)]
        ranked.sort(key=lambda item: (-item["finconrank_score"], item["original_rank"]))
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        output.append({"query_id": row["query_id"], "language": language, "source_report_group": row["group_id"],
                       "candidates": ranked, "checkpoint_path": str(args.adapter), "checkpoint_sha256": checkpoint_hash,
                       "physical_gpu_id": int(os.environ["T2_PHYSICAL_GPU_ID"]), "inference_timestamp": now(),
                       "protocol_sha256": sha(RUN / "01_PROTOCOL_AND_INPUTS/protocol_lock.json")})
        count += len(ranked)
        atomic(args.output.with_suffix(".partial.jsonl"), output, True)
        if index % 20 == 0 or index == len(records):
            print({"queries": index, "total": len(records), "pairs": count}, flush=True)
    atomic(args.output, output, True)
    atomic(args.stats, {"query_count": len(output), "scored_candidates": count, "seconds": time.perf_counter()-started,
                       "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3, "physical_gpu_id": int(os.environ["T2_PHYSICAL_GPU_ID"]),
                       "output_sha256": sha(args.output), "checkpoint_sha256": checkpoint_hash,
                       "finished_at": now(), "scoring_implementation": "original equivalent_last_token_lm_head_batch2", "qrels_read": False})


if __name__ == "__main__":
    main()
