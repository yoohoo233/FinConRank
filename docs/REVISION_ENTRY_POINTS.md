# Revision Entry Points And Actual Prerequisites

All entries are collected from completed evidence snapshots and matched against
their file manifests. No model, metric or statistical entry was executed for
packaging. Use new output directories, supplied legal data and existing local
models. Do not run these commands against frozen historical results.

## Historical Main Experiment Versus Revision Fusion

- `select_finconrank_strict_nested.py`: original main experiment; its historical
  four-score grid retains the `parts[2] > 0` condition (positive FinConRank weight).
- `run_major_revision_fusion_ablations.py`: separate deduplicated-v2 revision entry.
  Three-score search: 21 grid points. Zero-allowed four-score search: 56 grid points.
  Historical-positive comparison grid: 35 points. Scores, normalizers, Metric-Aware
  parameters and deployment selection retain their actual historical implementation.
- Inputs: frozen main run, inner OOF scores, original source candidates, allowed
  qrels, OCR/PDF evidence and per-fold config/provenance. None are bundled.
- Outputs: independently locked configs/predictions, metrics, paired deltas and
  the revision's own eight-test Holm family. They are not a retuned main result.

```bash
python scripts/run_major_revision_fusion_ablations.py --help
```

The CLI has lock/select/freeze/evaluate gates. Consult help and the source for
their required paths before use. Public Source Data contains the completed
outputs; running those stages is not required to review the numeric tables.

## Direct DocReRank Comparison

`run_major_revision_docrerank.py` uses a fixed **ColQwen2 top-20** pool for all
direct rerankers. It is not the main fused system's union/Mono candidate scope.
DocReRank identity: `DocReRank/DocReRank-Reranker`, revision
`292fcbdd8bb3ce066cd6c9a0de2d8212ccbd8d3a`. Its source model card recorded English
and CC-BY-4.0. The model is not bundled or relicensed by this code archive.
Its relevance prompt, two-token softmax, BF16 and 200704/589824 pixel limits are
retained. These differ from FinConRank's True-minus-False FP16 path; do not erase
those implementation boundaries or call the comparison equal-budget training.

```bash
python scripts/run_major_revision_docrerank.py --help
```

Inputs include an existing local DocReRank adapter/base, image roots, frozen
same-pool candidates and the comparison manifests. Output includes frozen direct
predictions, same-pool metrics, type subsets and an independent eight-test family.
Preparation/worker/freeze/evaluation stages must not be run during code review.

## Fixed-Corpus T2 Training Sensitivity

Actual archival runners live under `experiments/t2/06_REPRODUCIBILITY/scripts/`:

- `preflight_v2.py`, `t2_common.py`: input, image, recipe and source-hash gates.
- `scheduler_v2.py`, `scheduler_fixed1234.py`, `scheduler_fixed1245.py`: historical
  resumable queue variants. The final historical dispatcher is fixed1245.
- `launch_v2.py`, `launch_fixed1234.py`, `launch_fixed1245.py`: detached launchers;
  **importing or executing a launcher can start a process. Do not run --help on it.**
- `score_v2.py`: checkpoint scoring; `evaluate_v2.py`: prediction freeze and
  development-only reporting. `audit_complete_v2.py` / `verify_matrix_v2.py` are
  historical completed-output audit utilities, not current package tests.
- `test_v2.py` is the original synthetic plan/scheduler test. It has not been run
  here against real archival inputs. The actual completed job matrix reference is
  `experiments/t2/01_PROTOCOL_AND_INPUTS/job_manifest.reference.json`.

The unchanged numerical plan is in
`input/FinConRank_T2_fixed_corpus_execution_plan_20261013_v2.json`. Path values in
that copy are relative/templates, not a valid relocated frozen lock. It records
four local validation units, 20 training families and 28 fixed-checkpoint scores:
margin 0.1/0.2/0.4, LR 1e-5/2e-5/4e-5, reference trajectory epochs 2/3/4.
English 216 and Chinese 342 validation queries; these are not full-test results.
Epoch differences include curriculum exposure. There is no new significance
test, optimal-setting selection, page-exclusion claim or full-system robustness claim.

Historical execution order (archival prerequisites first, **not runnable from this
archive alone**): preflight_v2, then launch_fixed1245/scheduler_fixed1245; the queue
calls scoring, freezes all 28 outputs and invokes evaluate_v2. Launchers' integrity
guards are rebound to their released scheduler copies, not removed. Every numerical
recipe and CPU/GPU scheduling rule remains as in the archived scripts.

Missing archival prerequisites, intentionally not fabricated:

1. Original main source-run protocol and source/input/checkpoint hash manifests,
   per-unit training pairs and split/candidate files, source image roots and local
   base/Mono snapshots. No full training pairs, qrels or weights are included.
2. The exact historical page-audit ZIP referenced by the plan, plus its source-hash
   audit files. Only non-identifying Source Data summaries are public here.
3. `input/FinConRank_T2_fixed_corpus_Codex_CLI_goal_20261013_v2.md`, the historical
   private execution instruction read by preflight for provenance; it is excluded.
4. A newly validated relocation/lock procedure for a different environment.
   Original runtime source hashes cannot be silently treated as matching edited
   release copies. Existing integrity guards intentionally fail on mismatches.

These are explicit missing **runtime inputs/provenance**, not missing algorithms
or a claim that the historical models should be retrained during release review.

## Summary, Paired Statistics, Diagnostics And Costs

- `report_major_revision_results.py`, `verify_major_revision_query_metrics.py`:
  historical summary/verification; require frozen results and some internal stage
  metadata. Internal reviewer files/logs are intentionally excluded.
- `run_major_revision_cached_diagnostics.py`,
  `audit_major_revision_training_distribution.py`: existing cached precision,
  OCR coverage, automatic parser-field coverage and training-label distributions.
  Automatic parser fields are not independently adjudicated parser accuracy.
- `verify_major_revision_human_counts.py`: archival single-round quality counts;
  human annotations remain separate, are not copied into the code archive, and
  are not used as training or selection inputs.
- `run_major_revision_cost_benchmark.py`, `measure_major_revision_serial_cost.py`,
  `measure_major_revision_cpu_calibration.py`, `measure_major_revision_cpu_fusion.py`,
  `summarize_major_revision_cost.py`, queue/monitor helpers: actual historical
  measurements and aggregation. They require the original locked timing queries,
  local models and results; **do not execute them for code/package validation**.
- `evaluate_finconrank_strict_nested.py` and the official-metric correction retain
  the original clustered paired methods. Revision fusion/DocReRank each keep their
  separate families and score/pool definitions. No tests were newly run here.

See the separate Source Data README for units, directions, denominators and
statistical-family scope. Final v9 section and table references are recorded in docs/MANUSCRIPT_V9_MAP.md.
The separate Source Data archive contains MANUSCRIPT_TABLE_MAP.md.

## Historical Packaging Dependency

The archived queue also invokes `package_v2.py`, now included at the same relative
script location. It is an internal historical evidence packager, not a public
release filter: its outputs may include local logs and provenance records. Do not
upload its output without a separate disclosure/privacy review. This task did not
execute it; the supplied public Source Data archive was separately deidentified.
