"""Package completed T2 evidence, excluding model weights and page images."""

# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import csv
import hashlib
import re
import zipfile
from datetime import datetime, timezone
from t2_common import RUN, ROOT, read, sha, atomic, now


def main():
    checks = read(RUN / "05_RESOURCE_AND_QA/checks.json")
    if checks["training_jobs"] != 20 or checks["scoring_jobs"] != 28 or not checks["frozen_inputs_unchanged"]:
        raise RuntimeError("incomplete computation")
    result = (RUN / "04_RESULTS/metrics_by_language.csv").read_text()
    scope = "Fixed corpus, query-disjoint, known training-negative page/report exposure. This is not page-isolation repair or unseen-report evidence; local outer0 development sensitivity only. No new T1/H1 task, depth/Metric-Aware sensitivity, significance testing or best-parameter selection. Epoch comparisons include curriculum exposure differences."
    (RUN / "00_README_AND_INDEX/README_FIRST.md").write_text("# FinConRank T2 Fixed-Corpus Sensitivity v2\n\n"+scope+"\n\nRead results_summary.md, protocol_lock.json and checks.json. Models remain on server; weights are excluded.\n")
    (RUN / "00_README_AND_INDEX/results_summary.md").write_text("# Results\n\n"+scope+"\n\n20 training families and28 checkpoint scoring conditions completed. Validation denominator: English216/Chinese342. Full-precision CSV follows (not parameter selection):\n\n```csv\n"+result+"```\n")
    (RUN / "00_README_AND_INDEX/completion_audit.md").write_text("# Completion Audit\n\nAll20 training jobs and28 scoring jobs complete; all predictions frozen before validation labels read. Query/candidate coverage checked; official metrics cross-checked; input hashes unchanged. Original source/scripts, training pairs, blocked v1 and page audit preserved. No outer-test query scoring or outer-test qrels evaluation.\n\n"+scope+"\n")
    atomic(RUN / "00_README_AND_INDEX/status.json", {"status":"COMPLETE_COMPUTATION_PACKAGING_VERIFY", "training_complete":20,"scoring_complete":28,"updated_at":now(),"page_exclusion_not_claimed":True})
    excluded = (".safetensors", ".bin", ".pt", ".png", ".jpg", ".jpeg", ".pdf", ".partial.jsonl")
    files = [p for p in sorted(RUN.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and not str(p).endswith(excluded)
             and p.name not in ("evidence_file_manifest.csv", "sha256sums.txt", "scheduler.lock") and ".annotation_auth" not in p.name]
    entries = []
    for path in files:
        content = path.read_text(errors="replace")
        if re.search(r"hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY", content):
            raise RuntimeError(f"sensitive pattern {path}")
        entries.append({"relative_path":str(path.relative_to(RUN)),"source_path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)})
    manifest_path = RUN / "evidence_file_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    checksum = RUN / "sha256sums.txt"
    checksum.write_text("".join(f"{r['sha256']}  {r['relative_path']}\n" for r in entries))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = ROOT / "exports" / f"finconrank_t2_fixed_corpus_sensitivity_evidence_{stamp}.zip"
    with zipfile.ZipFile(zip_path,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest_path, checksum]:
            archive.write(path,str(path.relative_to(RUN)))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip():
            raise RuntimeError("corrupt ZIP")
        for entry in entries:
            value = archive.read(entry["relative_path"])
            if len(value)!=entry["bytes"] or hashlib.sha256(value).hexdigest()!=entry["sha256"]:
                raise RuntimeError("archive hash mismatch")
    atomic(RUN / "05_RESOURCE_AND_QA/zip_validation.json", {"zip_path":str(zip_path),"zip_bytes":zip_path.stat().st_size,"sha256":sha(zip_path),"entry_verification":True,"sensitive_scan_passed":True})
    atomic(RUN / "00_README_AND_INDEX/status.json", {"status":"COMPLETE", "training_complete":20,"scoring_complete":28,"updated_at":now(),"zip_path":str(zip_path),"page_exclusion_not_claimed":True})
    print({"zip":str(zip_path),"bytes":zip_path.stat().st_size},flush=True)


if __name__ == "__main__":
    main()
