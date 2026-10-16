"""Resumable 20-train/28-score fixed-corpus queue, maximum four idle GPUs."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import fcntl
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from t2_common import RUN, ROOT, read, rows, sha, atomic, csvfile, now, verify_protocol
from major_revision_gpu_scheduler import parse_inventory


def identity(pid, job):
    path = Path(f"/proc/{pid}/cmdline")
    return path.exists() and str(RUN) in path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def free(gpu):
    return gpu.index in {1,2,4,5} and gpu.process_visibility_known and not gpu.compute_pids and gpu.memory_used_mib < 1500 and gpu.utilization_percent < 5


def build_command(job, protocol, attempt_dir, jobs):
    unit = job["unit"]
    if job["kind"] == "train":
        family = job["family"]
        if sha(job["train_pairs"]) != job["pair_sha256"]:
            raise RuntimeError("training source changed")
        return [sys.executable, str(ROOT / "scripts/train_finconrank.py"), "--train-pairs", job["train_pairs"],
                "--dev-pairs", str(RUN / "01_PROTOCOL_AND_INPUTS/empty_dev_pairs.jsonl"), "--output-dir", str(attempt_dir / "model"),
                "--base-model", protocol["base_model"], "--init-adapter", protocol["initial_adapter"], "--variant", "full",
                "--epochs", str(family["train_epochs"]), "--margin", str(family["margin"]), "--learning-rate", str(family["learning_rate"]),
                "--batch-size", "2", "--gradient-accumulation", "2", "--max-pixels", "200704", "--seed", str(unit["training_seed"]), "--device", "cuda"]
    dependency = next(row for row in jobs if row["job_id"] == job["depends_on"])
    adapter = Path(dependency["completed_output"]) / f"epoch_{job['condition']['checkpoint_epoch']}_adapter"
    if sha(job["candidate_path"]) != job["candidate_sha256"]:
        raise RuntimeError("candidate source changed")
    job["checkpoint_sha256"] = sha(adapter / "adapter_model.safetensors")
    job["checkpoint_path"] = str(adapter)
    return [sys.executable, str(Path(__file__).with_name("score_v2.py")), "--candidates", job["candidate_path"],
            "--adapter", str(adapter), "--output", str(attempt_dir / "predictions.jsonl"), "--stats", str(attempt_dir / "stats.json")]


def completed(job):
    attempt = job["attempts"][-1]
    directory = Path(attempt["directory"])
    if job["kind"] == "train":
        model = directory / "model"
        summary = read(model / "training_summary.json")
        if summary["epochs"] != job["family"]["train_epochs"] or summary["seed"] != job["unit"]["training_seed"]:
            raise RuntimeError("training output recipe mismatch")
        expected_recipe = {"variant":"full", "learning_rate":job["family"]["learning_rate"], "margin":job["family"]["margin"],
                           "batch_size":2, "gradient_accumulation":2, "max_pixels":200704, "dev_pairs":0}
        if any(summary.get(key) != value for key,value in expected_recipe.items()):
            raise RuntimeError("training scientific configuration mismatch")
        configs = []
        for epoch in range(1, job["family"]["train_epochs"]+1):
            adapter = model / f"epoch_{epoch}_adapter"
            configs.append({"epoch": epoch, "path": str(adapter), "sha256": sha(adapter / "adapter_model.safetensors")})
        job["checkpoint_manifest"] = configs
        job["completed_output"] = str(model)
    else:
        path = directory / "predictions.jsonl"
        predictions = rows(path)
        expected = rows(job["candidate_path"])
        if [r["query_id"] for r in predictions] != [r["query_id"] for r in expected]:
            raise RuntimeError("scoring query coverage mismatch")
        for prediction, reference in zip(predictions, expected):
            if {r["corpus_id"] for r in prediction["candidates"]} != {r["corpus_id"] for r in reference["candidates"]}:
                raise RuntimeError("scoring candidate coverage mismatch")
            if len(prediction["candidates"]) != len(reference["candidates"]):
                raise RuntimeError("duplicate scoring candidates")
            if prediction["checkpoint_sha256"] != job["checkpoint_sha256"]:
                raise RuntimeError("wrong checkpoint provenance")
        job["completed_output"] = str(path)
        job["output_sha256"] = sha(path)


def main():
    log_root = RUN / "06_REPRODUCIBILITY/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    resource = RUN / "05_RESOURCE_AND_QA"
    resource.mkdir(parents=True, exist_ok=True)
    lock = (resource / "scheduler.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = verify_protocol()
    state_path = resource / "queue_state.json"
    jobs = read(state_path)["jobs"] if state_path.exists() else read(RUN / "01_PROTOCOL_AND_INPUTS/job_manifest.json")
    live = {}
    for job in jobs:
        if job["status"] == "RUNNING":
            attempt = job["attempts"][-1]
            if identity(attempt["pid"], job):
                live[job["job_id"]] = None
            else:
                try:
                    completed(job)
                    job["status"] = "COMPLETE"
                    attempt["status"] = "recovered_complete_after_interruption"
                except Exception:
                    job["status"] = "PLANNED" if len(job["attempts"]) < 2 else "WAITING_AUTHOR_INPUT"
                    attempt["status"] = "interrupted_missing_output"
        elif job["status"] == "COMPLETE":
            completed(job)
    atomic(resource / "scheduler_pid.json", {"pid": os.getpid(), "started_at": now(), "run": str(RUN), "resume_command": f"PYTHONDONTWRITEBYTECODE=1 {sys.executable} {__file__}"})
    observations = []
    last_status = 0
    gpu_log = (resource / "gpu_monitor.csv").open("a", buffering=1)
    if gpu_log.tell() == 0:
        gpu_log.write("timestamp,index,uuid,used_mib,free_mib,utilization,pids,visibility_known\n")
    while True:
        for job in jobs:
            if job["job_id"] not in live:
                continue
            process = live[job["job_id"]]
            attempt = job["attempts"][-1]
            code = process.poll() if process is not None else (None if identity(attempt["pid"], job) else -99)
            if code is None:
                continue
            del live[job["job_id"]]
            attempt.update({"finished_at": now(), "exit_code": code, "wall_seconds": time.time()-attempt["started_unix"]})
            try:
                if code not in (0, -99):
                    raise RuntimeError(f"worker exit {code}")
                completed(job)
                job["status"] = "COMPLETE"
                attempt["status"] = "COMPLETE"
            except Exception as error:
                attempt["status"] = "FAILED"
                attempt["error"] = str(error)
                job["status"] = "PLANNED" if len(job["attempts"]) < 2 else "WAITING_AUTHOR_INPUT"
                print({"failed_job": job["job_id"], "attempt": len(job["attempts"]), "error": str(error)}, flush=True)
        try:
            snapshot = parse_inventory()
            observations.append(snapshot)
            observations = observations[-3:]
            for gpu in snapshot:
                gpu_log.write(f"{now()},{gpu.index},{gpu.uuid},{gpu.memory_used_mib},{gpu.memory_free_mib},{gpu.utilization_percent},{'|'.join(map(str,gpu.compute_pids))},{gpu.process_visibility_known}\n")
        except Exception as error:
            snapshot = []
            observations = []
            print({"gpu_inventory_error": str(error)}, flush=True)
        active_cards = {job["attempts"][-1]["gpu_id"] for job in jobs if job["job_id"] in live}
        candidates = []
        if len(observations) == 3:
            for gpu in snapshot:
                if gpu.index not in active_cards and all(any(prev.uuid == gpu.uuid and free(prev) for prev in snaps) for snaps in observations):
                    candidates.append(gpu)
        ready = [job for job in jobs if job["status"] == "PLANNED" and (job["kind"] == "train" or any(dep["job_id"] == job["depends_on"] and dep["status"] == "COMPLETE" for dep in jobs))]
        ready.sort(key=lambda job: (job["kind"] != "score", job["job_id"]))
        available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        slots = max(0, min(4-len(active_cards), int((available_kib / 1024**2-16)/24)))
        for gpu, job in zip(candidates[:slots], ready):
            if not any(current.uuid == gpu.uuid and free(current) for current in parse_inventory()):
                continue
            attempt_number = len(job["attempts"])+1
            if attempt_number > 2:
                job["status"] = "WAITING_AUTHOR_INPUT"
                continue
            directory = RUN / ("02_TRAINING" if job["kind"] == "train" else "03_PREDICTIONS") / "tasks" / job["job_id"] / f"attempt_{attempt_number}"
            directory.mkdir(parents=True, exist_ok=False)
            command = build_command(job, protocol, directory, jobs)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu.index), T2_PHYSICAL_GPU_ID=str(gpu.index),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{ROOT}/src:{ROOT}/scripts", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_ENDPOINT="https://hf-mirror.com", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            logfile = log_root / f"{job['job_id']}_attempt{attempt_number}.log"
            with logfile.open("a") as handle:
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            attempt = {"attempt": attempt_number, "directory": str(directory), "gpu_id": gpu.index, "gpu_uuid": gpu.uuid,
                       "pid": process.pid, "command": command, "log_path": str(logfile), "started_at": now(), "started_unix": time.time(), "status": "RUNNING"}
            job["attempts"].append(attempt)
            job["status"] = "RUNNING"
            live[job["job_id"]] = process
            print({"started": job["job_id"], "gpu": gpu.index, "pid": process.pid}, flush=True)
        complete_train = sum(j["kind"] == "train" and j["status"] == "COMPLETE" for j in jobs)
        complete_score = sum(j["kind"] == "score" and j["status"] == "COMPLETE" for j in jobs)
        status = "GPU_RUNNING" if live else "WAITING_GPU" if ready else "WAITING_AUTHOR_INPUT"
        if complete_train == 20 and complete_score == 28:
            status = "ALL_PREDICTIONS_READY"
        atomic(state_path, {"updated_at": now(), "pid": os.getpid(), "status": status, "jobs": jobs})
        atomic(resource / "gpu_leases.json", {"updated_at": now(), "active": [j["attempts"][-1] for j in jobs if j["job_id"] in live], "maximum_cards": 4})
        if time.time()-last_status > 180 or status == "ALL_PREDICTIONS_READY":
            atomic(RUN / "00_README_AND_INDEX/status.json", {"status": status, "updated_at": now(), "training_complete": complete_train,
                   "scoring_complete": complete_score, "running_jobs": list(live), "execution_contract_pass": True, "page_exclusion_not_claimed": True})
            print({"status": status, "train_complete": complete_train, "score_complete": complete_score, "running": len(live)}, flush=True)
            last_status = time.time()
        if status == "ALL_PREDICTIONS_READY":
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_v2.py"))], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            if result.returncode:
                raise RuntimeError("post-inference CPU evaluation failed")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("package_v2.py"))], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            if result.returncode:
                raise RuntimeError("evidence package failed")
            break
        if not live and not ready:
            break
        time.sleep(10 if len(observations) < 3 or candidates else 60)


if __name__ == "__main__":
    main()
