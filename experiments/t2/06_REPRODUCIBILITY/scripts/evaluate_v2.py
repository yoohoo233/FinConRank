"""Freeze all 28 predictions before validation-only descriptive evaluation."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
from collections import defaultdict
from pathlib import Path
import statistics
import shutil
from t2_common import RUN, read, rows, sha, atomic, csvfile, verify_protocol, now
from correct_finconrank_strict_nested_official_metrics import evaluate_official, pytrec_check


def main():
    protocol = verify_protocol()
    queue = read(RUN / "05_RESOURCE_AND_QA/queue_state.json")
    jobs = queue["jobs"]
    if len([j for j in jobs if j["kind"] == "train" and j["status"] == "COMPLETE"]) != 20 or len([j for j in jobs if j["kind"] == "score" and j["status"] == "COMPLETE"]) != 28:
        raise RuntimeError("not all planned jobs complete; qrels must remain sealed")
    freeze_path = RUN / "03_PREDICTIONS/prediction_freeze_manifest.json"
    frozen = []
    training_manifest, epochs, resources = [], [], []
    for job in jobs:
        attempt = job["attempts"][-1]
        if job["kind"] == "train":
            model = Path(job["completed_output"])
            config = read(model / "training_config.json")
            training_manifest.append({"job_id": job["job_id"], "task_id": job["unit"]["task_id"], "family": job["family"]["family_id"],
                                      "seed": job["unit"]["training_seed"], "margin": config["margin"], "learning_rate": config["learning_rate"],
                                      "epochs": config["epochs"], "train_pairs": config["train_pairs"], "dev_pairs": config["dev_pairs"],
                                      "input_sha256": job["pair_sha256"], "checkpoint_manifest": str(job["checkpoint_manifest"]), "output_path": str(model)})
            summary = read(model / "training_summary.json")
            resources.append({"job_id": job["job_id"], "kind": "train", "physical_gpu_id": attempt["gpu_id"],
                              "seconds": summary["total_seconds"], "peak_vram_gib": summary["max_vram_gib"], "attempts": len(job["attempts"]),
                              "reserved_gpu_seconds_including_failed_attempts":sum(a.get("wall_seconds",0) for a in job["attempts"])})
            for record in read(model / "training_history.json"):
                epochs.append({"job_id": job["job_id"], **{k: str(v) if isinstance(v, (dict,list)) else v for k,v in record.items()}})
            for checkpoint in job["checkpoint_manifest"]:
                src = Path(checkpoint["path"]) / "adapter_config.json"
                dest = RUN / "02_TRAINING/adapter_config_snapshots" / f"{job['job_id']}_epoch{checkpoint['epoch']}.json"
                shutil.copy2(src, dest)
        else:
            source = Path(job["completed_output"])
            if sha(source) != job["output_sha256"]:
                raise RuntimeError("prediction changed before freeze")
            destination = RUN / "03_PREDICTIONS/validation_predictions" / f"{job['unit']['task_id']}__{job['condition']['condition_id']}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha(destination) != sha(source):
                raise RuntimeError("existing frozen destination mismatch")
            if not destination.exists():
                shutil.copy2(source, destination)
            frozen.append({"job_id": job["job_id"], "task_id": job["unit"]["task_id"], "language": job["unit"]["language"],
                           "condition": job["condition"]["condition_id"], "path": str(destination), "sha256": sha(destination),
                           "source_path": str(source), "checkpoint_path": job["checkpoint_path"], "checkpoint_sha256": job["checkpoint_sha256"]})
            stats = read(Path(attempt["directory"]) / "stats.json")
            resources.append({"job_id": job["job_id"], "kind": "score", "physical_gpu_id": attempt["gpu_id"],
                              "seconds": stats["seconds"], "peak_vram_gib": stats["peak_vram_gib"], "attempts": len(job["attempts"]),
                              "reserved_gpu_seconds_including_failed_attempts":sum(a.get("wall_seconds",0) for a in job["attempts"])})
    if freeze_path.exists():
        if read(freeze_path)["predictions"] != frozen:
            raise RuntimeError("freeze manifest changed")
    else:
        atomic(freeze_path, {"frozen_at": now(), "predictions": frozen, "validation_qrels_read": False,
                            "protocol_sha256": sha(RUN / "01_PROTOCOL_AND_INPUTS/protocol_lock.json")})
    # No label read is permitted above this point.
    source_run = Path(protocol["plan"]["source_run"])
    dev_qrels = {lang: read(source_run / f"sealed_qrels/{lang}_outer_0_development_qrels.json") for lang in ("en", "zh")}
    query_metrics = []
    max_error = 0.0
    for entry in frozen:
        for prediction in rows(entry["path"]):
            qid = prediction["query_id"]
            qrels = dev_qrels[entry["language"]][qid]
            metric = evaluate_official(prediction["candidates"], qrels)
            max_error = max(max_error, pytrec_check(prediction["candidates"], qrels, metric))
            query_metrics.append({"task_id": entry["task_id"], "language": entry["language"], "condition": entry["condition"],
                                  "query_id": qid, "report_group": prediction["source_report_group"], **metric})
    metrics = protocol["plan"]["reporting"]["metrics"]
    by_unit, by_language = [], []
    for level, output in (("task_id", by_unit), ("language", by_language)):
        buckets = defaultdict(list)
        for row in query_metrics:
            buckets[(row[level], row["condition"])].append(row)
        for (key, condition), values in sorted(buckets.items()):
            output.append({level: key, "language": values[0]["language"], "condition": condition, "query_count": len(values),
                           "report_group_count": len({r["report_group"] for r in values}),
                           **{m: statistics.fmean(r[m] for r in values) for m in metrics}})
    reference = {(r["task_id"],r["query_id"]): r for r in query_metrics if r["condition"] == "reference_epoch3"}
    deltas = [{"task_id": r["task_id"], "language": r["language"], "condition": r["condition"], "query_id": r["query_id"],
               **{f"delta_{m}": r[m]-reference[(r["task_id"],r["query_id"])][m] for m in metrics}} for r in query_metrics]
    for lang, expected in protocol["plan"]["reporting"]["expected_pooled_validation_queries"].items():
        for condition in protocol["plan"]["evaluated_conditions"]:
            subset = [r for r in query_metrics if r["language"] == lang and r["condition"] == condition["condition_id"]]
            if len(subset) != expected or len({r["query_id"] for r in subset}) != expected:
                raise RuntimeError("pooled validation denominator/uniqueness failure")
    if max_error > 1e-12:
        raise RuntimeError("official metric cross-check failed")
    ref_language = {r["language"]: r for r in by_language if r["condition"] == "reference_epoch3"}
    language_deltas = [{"language": r["language"], "condition": r["condition"], "query_count": r["query_count"],
                       **{f"delta_{m}": r[m]-ref_language[r["language"]][m] for m in metrics}} for r in by_language]
    for name, data in (("metrics_by_unit.csv", by_unit), ("metrics_by_language.csv", by_language), ("query_level_metrics.csv",query_metrics),
                       ("query_level_deltas.csv", deltas), ("deltas_vs_reference.csv", language_deltas)):
        csvfile(RUN / "04_RESULTS" / name, data)
    csvfile(RUN / "02_TRAINING/training_manifest.csv", training_manifest)
    csvfile(RUN / "02_TRAINING/epoch_history.csv", epochs)
    csvfile(RUN / "05_RESOURCE_AND_QA/resource_summary.csv", resources)
    manifest = [{"job_id":j["job_id"], "kind":j["kind"], **a} for j in jobs for a in j["attempts"]]
    csvfile(RUN / "05_RESOURCE_AND_QA/gpu_run_manifest.csv", [{k:str(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in manifest])
    with (RUN / "01_PROTOCOL_AND_INPUTS/input_manifest.csv").open() as handle:
        import csv
        rechecks = [{**r, "current_sha256":sha(r["path"])} for r in csv.DictReader(handle)]
    if any(r["sha256"] != r["current_sha256"] for r in rechecks):
        raise RuntimeError("frozen input changed")
    csvfile(RUN / "05_RESOURCE_AND_QA/frozen_input_hash_recheck.csv", rechecks)
    atomic(RUN / "05_RESOURCE_AND_QA/checks.json", {"training_jobs":20, "scoring_jobs":28, "prediction_freeze_preceded_label_read":True,
           "validation_queries":{"en":216,"zh":342}, "official_metric_max_error":max_error, "frozen_inputs_unchanged":True,
           "outer_test_qrels_read":False, "significance_tests_run":False, "execution_contract_pass":True, "page_exclusion_not_claimed":True})
    atomic(RUN / "00_README_AND_INDEX/status.json", {"status":"COMPUTATION_COMPLETE_PACKAGING_PENDING", "training_complete":20, "scoring_complete":28, "updated_at":now()})
    print({"evaluation_complete":True,"queries":{"en":216,"zh":342},"maximum_metric_error":max_error}, flush=True)


if __name__ == "__main__":
    main()
