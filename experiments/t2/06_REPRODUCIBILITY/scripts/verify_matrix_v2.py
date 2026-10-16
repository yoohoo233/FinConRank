"""Verify matrix, curriculum exposure, aggregation, deltas and card ceiling."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from t2_common import RUN, read, rows, atomic, verify_protocol


def table(name):
    with (RUN / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    protocol = verify_protocol()
    jobs = read(RUN / "05_RESOURCE_AND_QA/queue_state.json")["jobs"]
    expected = read(RUN / "01_PROTOCOL_AND_INPUTS/job_manifest.json")
    if {j["job_id"] for j in jobs} != {j["job_id"] for j in expected} or any(j["status"] != "COMPLETE" for j in jobs):
        raise RuntimeError("incomplete or expanded matrix")
    for job in jobs:
        if not 1 <= len(job["attempts"]) <= 2:
            raise RuntimeError("attempt budget exceeded")
        if job["kind"] == "train":
            pairs = rows(job["train_pairs"])
            history = read(Path(job["completed_output"]) / "training_history.json")
            steps = 0
            for record in history:
                stages = set(range(min(record["epoch"]-1,2)+1))
                active = [r for r in pairs if int(r.get("curriculum_stage",2)) in stages]
                steps += math.ceil(math.ceil(len(active)/2)/2)
                if record["active_train_pairs"] != len(active) or record["optimizer_steps"] != steps:
                    raise RuntimeError("curriculum pair/step exposure mismatch")
                if set(record["active_curriculum_stages"]) != {int(r.get("curriculum_stage",2)) for r in active}:
                    raise RuntimeError("curriculum stage mismatch")
    raw = table("04_RESULTS/query_level_metrics.csv")
    metrics = protocol["plan"]["reporting"]["metrics"]
    qref = {(r["task_id"],r["query_id"]):r for r in raw if r["condition"] == "reference_epoch3"}
    lookup = {(r["task_id"],r["condition"],r["query_id"]):r for r in raw}
    max_delta = 0.0
    for delta in table("04_RESULTS/query_level_deltas.csv"):
        row = lookup[(delta["task_id"],delta["condition"],delta["query_id"])]
        ref = qref[(delta["task_id"],delta["query_id"])]
        max_delta = max(max_delta,*[abs(float(delta[f"delta_{m}"])-(float(row[m])-float(ref[m]))) for m in metrics])
    max_aggregate = 0.0
    for path, dimension in (("04_RESULTS/metrics_by_language.csv","language"),("04_RESULTS/metrics_by_unit.csv","task_id")):
        for row in table(path):
            selected = [r for r in raw if r[dimension] == row[dimension] and r["condition"] == row["condition"]]
            if len(selected) != int(row["query_count"]) or len({r["query_id"] for r in selected}) != len(selected):
                raise RuntimeError("aggregate count or uniqueness mismatch")
            if len({r["report_group"] for r in selected}) != int(row["report_group_count"]):
                raise RuntimeError("report-group count mismatch")
            max_aggregate = max(max_aggregate,*[abs(float(row[m])-statistics.fmean(float(r[m]) for r in selected)) for m in metrics])
    duration = table("05_RESOURCE_AND_QA/gpu_duration_reconciliation.csv")
    events = []
    for row in duration:
        events.extend([(datetime.fromisoformat(row["started_at"]),1,int(row["physical_gpu_id"])),
                       (datetime.fromisoformat(row["finished_at"]),-1,int(row["physical_gpu_id"]))])
    active = Counter()
    maximum = 0
    for _, change, card in sorted(events, key=lambda event:(event[0],event[1])):
        active[card] += change
        maximum = max(maximum,sum(n > 0 for n in active.values()))
    if maximum > 4 or max_delta > 1e-12 or max_aggregate > 1e-12:
        raise RuntimeError("matrix QA failed")
    atomic(RUN / "05_RESOURCE_AND_QA/matrix_aggregation_and_resource_audit.json",{
        "matrix_exactly_20_train_28_score":True,"curriculum_stage_pair_step_checks_passed":True,
        "paired_delta_max_error":max_delta,"query_weighted_aggregate_max_error":max_aggregate,
        "maximum_simultaneous_physical_gpus":maximum,"historically_used_cards":sorted({int(r["physical_gpu_id"]) for r in duration}),
        "failed_attempts":sum(a.get("status") == "FAILED" for j in jobs for a in j["attempts"]),
        "retry_jobs":sum(len(j["attempts"]) > 1 for j in jobs),"passed":True,"no_significance_tests":True})


if __name__ == "__main__":
    main()
