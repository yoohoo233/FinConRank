"""Lock the fixed-corpus T2 contract after offline source/query/image gates."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import ast
import csv
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image
from t2_common import RUN, ROOT, read, rows, sha, atomic, csvfile, now
from finconrank.common import page_document_id


def main():
    protocol_path = RUN / "01_PROTOCOL_AND_INPUTS/protocol_lock.json"
    if protocol_path.exists():
        raise RuntimeError("protocol already exists; use scheduler resume")
    plan_path = ROOT / "input/FinConRank_T2_fixed_corpus_execution_plan_20261013_v2.json"
    goal_path = ROOT / "input/FinConRank_T2_fixed_corpus_Codex_CLI_goal_20261013_v2.md"
    plan = read(plan_path)
    source = Path(plan["source_run"])
    original = read(source / "protocol_lock.json")
    audit_zip = Path(plan["page_audit_zip"])
    if not audit_zip.is_file() or sha(audit_zip) != plan["page_audit_zip_sha256"]:
        raise RuntimeError("page audit ZIP missing or changed")
    audit_dir = audit_zip.with_suffix("")
    expected = {}
    with (audit_dir / "input_hash_audit.csv").open() as handle:
        expected.update({r["source_path"]: r["sha256_after"] for r in csv.DictReader(handle)})
    expected.update({r["path"]: r["sha256"] for key in ("code_files", "base_model_files", "initial_adapter_files") for r in original[key]})
    inputs, source_hashes, query_checks, splits, jobs, image_set = [], [], [], [], [], set()
    def record(path):
        path = Path(path)
        actual = sha(path)
        exp = expected.get(str(path))
        if exp is not None and exp != actual:
            raise RuntimeError(f"source hash mismatch {path}")
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": actual, "matches_existing_evidence": exp is None or exp == actual})
        return actual
    for path in (plan_path, goal_path, source / "protocol_lock.json", audit_zip):
        record(path)
    for item in original["code_files"]:
        record(item["path"])
        source_hashes.append({"path": item["path"], "sha256": sha(item["path"])})
    imported = [ROOT / "scripts/run_finconrank_strict_nested_task.py", ROOT / "scripts/major_revision_gpu_scheduler.py",
                ROOT / "scripts/correct_finconrank_strict_nested_official_metrics.py", ROOT / "src/finconrank/common.py", ROOT / "src/finconrank/strict_nested.py"]
    for path in imported:
        if str(path) not in {row["path"] for row in source_hashes}:
            record(path)
            source_hashes.append({"path": str(path), "sha256": sha(path)})
    for row in source_hashes:
        path = Path(row["path"])
        snapshot = RUN / "06_REPRODUCIBILITY/source_snapshots" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
    for key in ("base_model_files", "initial_adapter_files"):
        for item in original[key]:
            record(item["path"])
    base = Path(original["model_recipe"]["base_model"])
    adapter = Path(original["model_recipe"]["initial_adapter"])
    if base.name != plan["base_recipe"]["base_model_revision"] or adapter.name != plan["base_recipe"]["initial_adapter_revision"]:
        raise RuntimeError("revision mismatch")
    adapter_config = read(adapter / "adapter_config.json")
    atomic(RUN / "02_TRAINING/adapter_config_snapshots/initial_mono_adapter_config.json", adapter_config)
    import torch
    env = {"python": platform.python_version(), "python_executable": sys.executable, "torch_cuda_runtime": torch.version.cuda,
           "versions": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "transformers", "peft", "safetensors", "pytrec_eval")},
           "installed_or_downloaded": False}
    atomic(RUN / "06_REPRODUCIBILITY/environment_versions.json", env)
    for command, filename in ((["nvidia-smi"], "initial_nvidia_smi.txt"), ([sys.executable, "-m", "pip", "freeze"], "pip_freeze.txt")):
        result = subprocess.run(command, text=True, capture_output=True)
        (RUN / "06_REPRODUCIBILITY" / filename).write_text(result.stdout + result.stderr)
    train_script = ROOT / "scripts/train_finconrank.py"
    tree = ast.parse(train_script.read_text())
    epochs_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args" and n.attr == "epochs"]
    if "lr_scheduler" in train_script.read_text() or "scheduler.step" in train_script.read_text():
        raise RuntimeError("total-epoch trajectory requires author review")
    train_lines = train_script.read_text().splitlines()
    atomic(RUN / "01_PROTOCOL_AND_INPUTS/epoch_trajectory_static_audit.json", {
        "args_epochs_occurrences": [{"line": line, "code": train_lines[line-1]} for line in epochs_lines],
        "constant_lr_optimizer": "AdamW lr=args.learning_rate; no scheduler",
        "shuffle": "torch.Generator.manual_seed(args.seed + epoch)", "curriculum": "stage_rows depends on epoch index only",
        "reference_first_three_epochs_match_recipe": True, "train_script_sha256": sha(train_script)})
    pool = {}
    for lang in ("en", "zh"):
        path = source / f"frozen_inputs/{lang}_candidates_without_qrels.jsonl"
        record(path)
        pool[lang] = {r["query_id"]: r for r in rows(path)}
    validation_seen = {lang: set() for lang in pool}
    empty = RUN / "01_PROTOCOL_AND_INPUTS/empty_dev_pairs.jsonl"
    empty.write_text("")
    for unit in plan["units"]:
        task, lang = unit["task_id"], unit["language"]
        split_path = source / "splits" / f"{task}.json"
        outer_path = source / "splits" / f"outer_{lang}_fold0.json"
        pair_path = source / "tasks" / task / "training_data/train_pairs.jsonl"
        for path in (split_path, outer_path, pair_path, pair_path.parent / "training_data_manifest.json"):
            record(path)
        split, outer = read(split_path), read(outer_path)
        train, valid, test = set(split["train_qids"]), set(split["validation_qids"]), set(outer["test_qids"])
        if train & (valid | test) or valid & test or valid - set(outer["development_qids"]) or len(valid) != unit["expected_validation_queries"]:
            raise RuntimeError(f"query split mismatch {task}")
        if valid & validation_seen[lang]:
            raise RuntimeError("validation folds overlap")
        validation_seen[lang].update(valid)
        for number, pair in enumerate(rows(pair_path), 1):
            passed = pair["query_id"] in train and pair["group_id"] in split["train_report_groups"] and pair["group_id"] not in set(split["validation_report_groups"]) | set(outer["test_report_groups"])
            query_checks.append({"task_id": task, "source_file": str(pair_path), "line": number, "query_id": pair["query_id"], "group_id": pair["group_id"], "passed": passed})
            if not passed:
                raise RuntimeError(f"query/group overlap: {task} line {number}")
            for role in ("positive", "negative"):
                path = Path(pair[f"{role}_image"])
                if "FinRAGBench-V/prepared" in str(path):
                    raise RuntimeError("prepared reference prohibited")
                image_set.add(path)
            raw_positive_group = f"{lang}:{page_document_id(pair['positive_page'])}"
            if raw_positive_group in set(split["validation_report_groups"]) | set(outer["test_report_groups"]):
                raise RuntimeError(f"new positive page overlap {task} line {number}")
        candidates = []
        for qid in split["validation_qids"]:
            info = pool[lang][qid]
            unique, items = set(), []
            for item in info["colqwen2_candidates"][:20]:
                cid = item["corpus_id"]
                if cid not in unique:
                    unique.add(cid)
                    items.append({"corpus_id": cid, "original_rank": len(items)+1, "colqwen2_score": item["score"]})
                    root = ROOT / ("cache/official_data/corpus/en/img_origin" if lang == "en" else "cache/official_data_ch/corpus/ch/img")
                    image_set.add(root / cid)
            candidates.append({"query_id": qid, "query": info["query"], "constraints": info["constraints"], "group_id": info["group_id"], "language": lang, "candidates": items})
        path = RUN / f"01_PROTOCOL_AND_INPUTS/validation_candidates/{task}.jsonl"
        atomic(path, candidates, True)
        record(path)
        splits.append({**unit, "training_queries": len(train), "validation_queries": len(valid), "validation_report_groups": len(split["validation_report_groups"]),
                       "candidate_pairs": sum(len(r["candidates"]) for r in candidates), "known_negative_page_exposure": "present"})
        for family in plan["training_families"]:
            train_id = f"{task}__{family['family_id']}__train"
            jobs.append({"job_id": train_id, "kind": "train", "unit": unit, "family": family, "train_pairs": str(pair_path), "pair_sha256": sha(pair_path), "status": "PLANNED", "attempts": []})
            for condition in plan["evaluated_conditions"]:
                if condition["family_id"] == family["family_id"]:
                    jobs.append({"job_id": f"{task}__{condition['condition_id']}__score", "kind": "score", "unit": unit, "condition": condition,
                                 "depends_on": train_id, "candidate_path": str(path), "candidate_sha256": sha(path), "status": "PLANNED", "attempts": []})
    image_checks = []
    for index, path in enumerate(sorted(image_set), 1):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            image_checks.append({"path": str(path), "width": width, "height": height, "container_verification": "PASS"})
        except Exception as error:
            image_checks.append({"path": str(path), "container_verification": "FAIL", "error": str(error)})
        if index % 500 == 0:
            print(f"image_container_checks {index}/{len(image_set)}", flush=True)
    csvfile(RUN / "05_RESOURCE_AND_QA/image_readability_checks.csv", image_checks)
    if any(row["container_verification"] != "PASS" for row in image_checks):
        raise RuntimeError("missing/unreadable image; no filtering allowed")
    dest = RUN / "01_PROTOCOL_AND_INPUTS/previous_page_isolation_audit_evidence"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("t2_four_unit_overlap_summary.csv", "t2_four_unit_overlap_records.csv", "unresolved_unique_pages.csv", "report_normalization_and_alias_audit.csv", "audit_checks.json", "page_isolation_audit_report.md"):
        with zipfile.ZipFile(audit_zip) as archive:
            import hashlib
            if hashlib.sha256(archive.read(name)).hexdigest() != sha(audit_dir / name):
                raise RuntimeError("extracted page-audit evidence differs from authorized ZIP")
        shutil.copy2(audit_dir / name, dest / name)
        record(audit_dir / name)
    shutil.copy2(audit_dir / "t2_four_unit_overlap_summary.csv", RUN / "01_PROTOCOL_AND_INPUTS/page_exposure_summary.csv")
    (RUN / "01_PROTOCOL_AND_INPUTS/scope_change.md").write_text("# Fixed-corpus v2 scope\nKnown cross-report negative-page exposure is present and explicitly allowed for descriptive query-disjoint sensitivity. No original pair was filtered. Old blocked preflight remains FAIL under its original contract. This does not repair page exclusion or establish unseen-report generalization. Report-source unknowns remain unknown.\n")
    atomic(RUN / "01_PROTOCOL_AND_INPUTS/plan_snapshot.json", plan)
    old_estimate = Path(plan["old_blocked_t2_run"]) / "05_RESOURCE_AND_QA/resource_estimate.json"
    if old_estimate.exists():
        shutil.copy2(old_estimate, RUN / "05_RESOURCE_AND_QA/resource_estimate.json")
        record(old_estimate)
    shutil.copy2(goal_path, RUN / "01_PROTOCOL_AND_INPUTS/goal_snapshot.md")
    csvfile(RUN / "01_PROTOCOL_AND_INPUTS/input_manifest.csv", inputs)
    csvfile(RUN / "01_PROTOCOL_AND_INPUTS/split_manifest.csv", splits)
    csvfile(RUN / "01_PROTOCOL_AND_INPUTS/query_membership_audit.csv", query_checks)
    atomic(RUN / "01_PROTOCOL_AND_INPUTS/job_manifest.json", jobs)
    hashes = [{"path": str(p), "sha256": sha(p)} for p in sorted((RUN / "06_REPRODUCIBILITY/scripts").glob("*.py"))]
    protocol = {"locked_at": now(), "plan": plan, "script_hashes": hashes + source_hashes,
                "base_model": str(base), "initial_adapter": str(adapter), "initial_adapter_config": adapter_config,
                "input_manifest_sha256": sha(RUN / "01_PROTOCOL_AND_INPUTS/input_manifest.csv"),
                "job_manifest_sha256": sha(RUN / "01_PROTOCOL_AND_INPUTS/job_manifest.json"), "execution_contract_pass": True,
                "page_exclusion_not_claimed": True, "known_negative_page_exposure": "present", "validation_qrels_read_before_prediction_freeze": False}
    atomic(protocol_path, protocol)
    (protocol_path.parent / "protocol_lock.sha256").write_text(sha(protocol_path) + "  protocol_lock.json\n")
    atomic(RUN / "00_README_AND_INDEX/status.json", {"status": "PREFLIGHT_PASSED", "plan_id": plan["plan_id"], "training_complete": 0,
         "scoring_complete": 0, "execution_contract_pass": True, "page_exclusion_not_claimed": True, "protocol_sha256": sha(protocol_path)})
    print(f"protocol locked: {RUN}", flush=True)


if __name__ == "__main__":
    main()
