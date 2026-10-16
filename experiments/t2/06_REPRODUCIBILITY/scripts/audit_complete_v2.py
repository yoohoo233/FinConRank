"""Independent final QA; run only after all frozen validation predictions exist."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import csv
import math
from datetime import datetime
from pathlib import Path
from t2_common import RUN, read, rows, sha, atomic, csvfile, verify_protocol
from correct_finconrank_strict_nested_official_metrics import evaluate_official, pytrec_check


def main():
    protocol = verify_protocol()
    if sha(RUN / "01_PROTOCOL_AND_INPUTS/input_manifest.csv") != protocol["input_manifest_sha256"] or sha(RUN / "01_PROTOCOL_AND_INPUTS/job_manifest.json") != protocol["job_manifest_sha256"]:
        raise RuntimeError("locked input/job manifest changed")
    freeze = read(RUN / "03_PREDICTIONS/prediction_freeze_manifest.json")
    queue = read(RUN / "05_RESOURCE_AND_QA/queue_state.json")
    jobs = queue["jobs"]
    train_jobs = [j for j in jobs if j["kind"] == "train"]
    score_jobs = [j for j in jobs if j["kind"] == "score"]
    if len(train_jobs) != 20 or len(score_jobs) != 28 or any(j["status"] != "COMPLETE" for j in jobs):
        raise RuntimeError("incomplete job matrix")
    if len(freeze["predictions"]) != 28:
        raise RuntimeError("incomplete prediction freeze")
    durations, recipe_checks = [], []
    for job in train_jobs:
        config = read(Path(job["completed_output"]) / "training_config.json")
        family = job["family"]
        expected = {"seed":job["unit"]["training_seed"],"margin":family["margin"],"learning_rate":family["learning_rate"],
                    "epochs":family["train_epochs"],"batch_size":2,"gradient_accumulation":2,"max_pixels":200704,"dev_pairs":0,
                    "variant":"full","curriculum":True,"use_constraints":True}
        if any(config.get(k) != v for k,v in expected.items()) or config["trainable_parameters"] <= 0:
            raise RuntimeError("training recipe mismatch")
        history = read(Path(job["completed_output"]) / "training_history.json")
        if [r["epoch"] for r in history] != list(range(1,family["train_epochs"]+1)):
            raise RuntimeError("epoch sequence mismatch")
        if any(not math.isfinite(r["mean_train_loss"]) or r["active_train_pairs"] <= 0 or r["optimizer_steps"] <= 0 for r in history):
            raise RuntimeError("training history lacks finite full epochs")
        if sha(job["train_pairs"]) != job["pair_sha256"]:
            raise RuntimeError("training pair source changed")
        recipe_checks.append({"job_id":job["job_id"],"recipe_match":True,"finite_epoch_losses":True,"trainable_parameters":config["trainable_parameters"]})
    for job in jobs:
        for attempt in job["attempts"]:
            end = attempt.get("finished_at")
            method = "dispatcher_start_finish"
            if end is None and job["kind"] == "score":
                stats_path = Path(attempt["directory"]) / "stats.json"
                if stats_path.exists():
                    end = read(stats_path)["finished_at"]
                    method = "worker_stats_finish_after_dispatcher_handover"
            seconds = (datetime.fromisoformat(end)-datetime.fromisoformat(attempt["started_at"])).total_seconds() if end else attempt.get("wall_seconds")
            if seconds is None:
                raise RuntimeError("unresolved attempt duration")
            durations.append({"job_id":job["job_id"],"kind":job["kind"],"attempt":attempt["attempt"],"physical_gpu_id":attempt["gpu_id"],
                              "gpu_uuid":attempt["gpu_uuid"],"started_at":attempt["started_at"],"finished_at":end,
                              "reserved_gpu_seconds":seconds,"duration_provenance":method,"exit_code":attempt.get("exit_code","UNAVAILABLE_AFTER_PARENT_HANDOVER")})
    candidate_source = {}
    for unit in protocol["plan"]["units"]:
        path = RUN / "01_PROTOCOL_AND_INPUTS/validation_candidates" / f"{unit['task_id']}.jsonl"
        candidate_source[unit["task_id"]] = {r["query_id"]:r for r in rows(path)}
    source = Path(protocol["plan"]["source_run"])
    # Label access is below all-complete and frozen-file gates only.
    for prediction in freeze["predictions"]:
        if sha(prediction["path"]) != prediction["sha256"]:
            raise RuntimeError("frozen prediction changed")
    dev = {lang:read(source / f"sealed_qrels/{lang}_outer_0_development_qrels.json") for lang in ("en","zh")}
    with (RUN / "04_RESULTS/query_level_metrics.csv").open() as handle:
        exported = {(r["task_id"],r["condition"],r["query_id"]):r for r in csv.DictReader(handle)}
    maximum = 0.0
    coverage = []
    for entry in freeze["predictions"]:
        reference = candidate_source[entry["task_id"]]
        predictions = rows(entry["path"])
        if sha(Path(entry["checkpoint_path"]) / "adapter_model.safetensors") != entry["checkpoint_sha256"]:
            raise RuntimeError("checkpoint bytes changed")
        if len(predictions) != len(reference) or {r["query_id"] for r in predictions} != set(reference):
            raise RuntimeError("validation query coverage mismatch")
        for row in predictions:
            candidates = row["candidates"]
            expected_ids = [r["corpus_id"] for r in reference[row["query_id"]]["candidates"]]
            if len(candidates) != len(expected_ids) or {r["corpus_id"] for r in candidates} != set(expected_ids):
                raise RuntimeError("candidate coverage mismatch")
            if row["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                raise RuntimeError("checkpoint mismatch")
            if any(not math.isfinite(r["finconrank_score"]) for r in candidates) or any("qrels" in r or "is_relevant" in r for r in [row,*candidates]):
                raise RuntimeError("non-finite score or leaked label field")
            metrics = evaluate_official(candidates,dev[entry["language"]][row["query_id"]])
            maximum = max(maximum,pytrec_check(candidates,dev[entry["language"]][row["query_id"]],metrics))
            previous = exported[(entry["task_id"],entry["condition"],row["query_id"])]
            maximum = max(maximum,*[abs(value-float(previous[name])) for name,value in metrics.items()])
        coverage.append({"task_id":entry["task_id"],"condition":entry["condition"],"query_count":len(predictions),"candidate_count":sum(len(r["candidates"]) for r in predictions),"passed":True})
    if maximum > 1e-12:
        raise RuntimeError("independent metric recomputation failed")
    csvfile(RUN / "05_RESOURCE_AND_QA/independent_recipe_checks.csv",recipe_checks)
    csvfile(RUN / "05_RESOURCE_AND_QA/independent_candidate_coverage.csv",coverage)
    csvfile(RUN / "05_RESOURCE_AND_QA/gpu_duration_reconciliation.csv",durations)
    atomic(RUN / "05_RESOURCE_AND_QA/independent_completion_audit.json",{"training_jobs":20,"score_jobs":28,"all_recipes_checked":True,
           "all_frozen_predictions_checked":True,"metric_max_absolute_error":maximum,
           "reserved_gpu_hours_including_all_attempts":sum(r["reserved_gpu_seconds"] for r in durations)/3600,
           "gpu_time_definition":"leased-process wall time including loading and CPU preprocessing, not CUDA kernel-only time",
           "unknown_exit_codes_kept_explicit":True,"outer_test_qrels_read":False,"passed":True,"page_exclusion_not_claimed":True})


if __name__ == "__main__":
    main()
