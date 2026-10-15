# Execution Order (Templates Only; Not Run During Packaging)

Every command below requires user-supplied inputs. Use a new output directory.
Read each CLI's help before use; paths in angle brackets are placeholders.
No command in this document was used to run a scientific experiment for packaging.

## 1. Data And Original Retrieval Inputs

Obtain FinRAGBench-V from the benchmark authors using their published instructions
at https://github.com/zhaosuifeng/FinRAGBench-V (retained from existing project
documentation; not fetched or reverified during this task);
this package contains no dataset or external official retrieval toolkit. Point
FINRAGBENCH_DATA_ROOT to the original corpus/queries/qrels tree, read-only.

```bash
python scripts/prepare_official_finragbench_inputs.py --language en --out-root <derived-en-directory>
python scripts/prepare_official_finragbench_inputs.py --language ch --out-root <derived-ch-directory>
python scripts/score_colqwen2_embeddings.py --help
python scripts/evaluate_and_export_retrieval.py --help
python scripts/rerank_with_monoqwen2vl.py --help
```

After separately generating official embeddings, illustrative scoring/export and
Mono commands are (same schema for each language):

```bash
CUDA_VISIBLE_DEVICES=<authorized-id> python scripts/score_colqwen2_embeddings.py score-worker \
  --run-dir <official-embedding-dir> --device cuda --worker-index 0 --num-workers 1 \
  --partial-out <partial-score-file> --topk 100
python scripts/score_colqwen2_embeddings.py merge --partials <partial-score-file> \
  --trec-out <tab-safe.trec> --topk 100
python scripts/evaluate_and_export_retrieval.py --trec <tab-safe.trec> \
  --qrels <original-qrels.tsv> --queries <queries.parquet> \
  --metrics-out <metrics.json> --predictions-out <predictions.jsonl> --trec-format tab-safe
CUDA_VISIBLE_DEVICES=<authorized-id> python scripts/rerank_with_monoqwen2vl.py worker \
  --predictions <predictions.jsonl> --image-root <language-image-root> \
  --base-model "$FINCONRANK_BASE_MODEL" --adapter "$FINCONRANK_INITIAL_ADAPTER" \
  --worker-index 0 --num-workers 1 --output <mono-worker.jsonl>
```

These examples do not assert that independently created candidate pools match a
historical main experiment. Retain candidate-depth provenance and merge workers
using the original merge subcommand when needed.

Preprocessing needs original archives plus pandas/pyarrow/Pillow. Embedding generation
requires the separately obtained official retrieval toolkit and a local ColQwen2
model. Late-interaction scoring and Mono inference require the resulting embeddings
or prediction inputs and a GPU. TREC writers/readers default to tab-safe IDs; legacy
whitespace repair is a historical recovery tool, not the new-data format.

## 2. Evidence Inputs

```bash
python scripts/build_real_multigranularity_index.py --help
python scripts/build_tesseract_ocr_index.py --help
python scripts/merge_ocr_pdf_multigranularity_index.py --help
python scripts/run_real_router_multigranularity.py --help
```

```bash
python scripts/build_real_multigranularity_index.py --out-dir <pdf-index> \
  --data-root "$FINRAGBENCH_DATA_ROOT" --en-image-root "$FINCONRANK_EN_IMAGES" \
  --zh-image-root "$FINCONRANK_ZH_IMAGES"
python scripts/build_tesseract_ocr_index.py --language en \
  --image-root "$FINCONRANK_EN_IMAGES" --out-dir <ocr-index>
python scripts/build_tesseract_ocr_index.py --language zh \
  --image-root "$FINCONRANK_ZH_IMAGES" --out-dir <ocr-index>
python scripts/merge_ocr_pdf_multigranularity_index.py --pdf-index-dir <pdf-index> \
  --ocr-index-dir <ocr-index> --out-dir <merged-index>
python scripts/run_real_router_multigranularity.py --index-dir <merged-index> \
  --out-dir <router-features> --strong-base-dir <baseline-dir>
```

These original builders require source PDF archives/images; OCR also requires
Tesseract and appropriate language packs. Neither tool nor data is installed here.
Merged index files are `{en,zh}_pages.jsonl` and `{en,zh}_blocks.jsonl`. Page rows
use corpus_id/title/text; block rows use corpus_id/text and any available geometry.
Router prediction rows provide per-candidate source_scores (text/block). These
are evidence features, not independent positive performance claims.

## 3. Constraints, Groups And Negatives

```bash
python scripts/run_finconrank_audit.py --help
python scripts/prepare_finconrank_oof.py --help
python scripts/build_finconrank_training_data.py --help
```

```bash
python scripts/run_finconrank_audit.py --run-dir <new-source-audit> \
  --baseline-dir <baseline-dir> --metric-dir <existing-metric-aware-inputs> \
  --index-dir <merged-index> --seed 42
python scripts/build_finconrank_training_data.py --run-dir <new-source-audit> \
  --training-dir <new-training-pairs> --baseline-dir <baseline-dir> \
  --index-dir <merged-index> --en-image-root "$FINCONRANK_EN_IMAGES" \
  --zh-image-root "$FINCONRANK_ZH_IMAGES" --seed 42
python scripts/prepare_finconrank_oof.py --run-dir <new-source-audit> \
  --all-pairs-dir <all-allowed-pairs> --output-dir <outer-fold-manifests> --folds 5 --seed 42
```

`<all-allowed-pairs>` must have train_pairs/dev_pairs/test_pairs filenames as
expected by the historical manifest builder. Do not infer that historical
test-pair creation permits their use in a nested training fold. For the nested
workflow, its task-specific development-qrels construction is the actual training
source. The historical audit's `--metric-dir` option is recorded as metadata only;
that entry does not read files from it. Original fold-manifest provenance is a
separate prerequisite for exact historical reconstruction, not provided by a template.

The audit entry creates the original query constraints and source grouping metadata
from supplied candidates/qrels; grouping uses connected report identifiers.
prepare_finconrank_oof builds deterministic outer group manifests from those inputs.
Actual nested tasks construct negatives only from their allowed development qrels.
Standalone historical pair builders are not permission to train on held-out qrels.

Typical prediction row: query_id, query, candidates with corpus_id/rank/score,
and only where permitted qrels. A constraint row contains query_id/group_id/
query_type/constraints. Pair rows contain query, constraints, positive_image,
negative_image, curriculum_stage and negative_types. The last two are sampling/
curriculum metadata, not relevance-label features in the query prompt.

The nested input contract in `finconrank.strict_nested.source_paths()` expects:

- BASELINE_DIR: `colqwen2_en_predictions.jsonl` and `colqwen2_zh_predictions.jsonl`.
- SOURCE_RUN: `query_constraints.jsonl` plus
  `mono_top50/en_monoqwen2vl_top50_predictions.jsonl` and
  `mono_top50/zh_monoqwen2vl_top50_predictions.jsonl`.
- SOURCE_OOF: `oof_manifest.json` with the original connected-report query folds.
- INDEX_DIR: `{en,zh}_pages.jsonl` and `{en,zh}_blocks.jsonl`.
- ROUTER_DIR: `{en,zh}_query_router_multi_granularity_mono_predictions.jsonl`.

The illustrative Mono command above retains its depth-20 default. Creating the
separate `mono_top50` inputs requires an adequate candidate pool and explicit
`--top-n 50 --output-depth 50`, not renaming depth-20 outputs. The nested worker
then scores the union of ColQwen2 top-20 and Mono top-20 candidates. These inputs
must be generated/verified independently; none are included in the code archive.

## 4. Standalone Adapter Training And Inference (GPU)

```bash
CUDA_VISIBLE_DEVICES=<one-authorized-id> python scripts/train_finconrank.py \
  --train-pairs <allowed-train-pairs.jsonl> --dev-pairs <allowed-dev-pairs.jsonl> \
  --output-dir <new-checkpoint-dir> --base-model "$FINCONRANK_BASE_MODEL" \
  --init-adapter "$FINCONRANK_INITIAL_ADAPTER" --variant full --epochs 3 \
  --learning-rate 0.00002 --margin 0.2 --batch-size 2 --gradient-accumulation 2 \
  --max-pixels 200704 --seed 42
python scripts/infer_finconrank.py --help
```

The seed template above is the original base recipe, not a replacement for each
nested task's deterministic language/fold seed. Standalone inference accepts
predictions, constraint records, image root and one adapter. For the released
models use the author-provided pinned repository/revision from README and select
the proper en/outer_N or zh/outer_N folder; do not add Mono on top again.
The standalone inference script retains its historical default depth 30 and
max_pixels 200704. These are not the nested worker's depth-20 union and
max_pixels 602112. Do not silently mix these configurations. The main nested
workflow below uses the actual dedicated worker and its unchanged recipe.

## 5. Nested Query-Group Training And Model Selection (GPU)

Set all documented input locations first. The shared corpus remains shared.

```bash
python scripts/prepare_finconrank_strict_nested.py lock --run-dir <fresh-nested-run>
python scripts/prepare_finconrank_strict_nested.py prepare --run-dir <fresh-nested-run>
python scripts/finconrank_strict_nested_scheduler_limited.py \
  --run-dir <fresh-nested-run> --max-concurrent-tasks 1 --allowed-gpus <authorized-id>
```

This is the original 5 outer x 4 inner query-group workflow. Inner validation
scores come from inner-trained models. After inner selection is frozen, outer
models train on outer development queries, then score their held-out query groups.
The scheduler calls run_finconrank_strict_nested_task and
select_finconrank_strict_nested. Calibration/fusion selection occurs only within
each outer development set; historical global assembled scores must not replace
the inner cross-fitted scores. Group labels do not prove document-page non-exposure.

## 6. Frozen Evaluation And Statistics (CPU, Not Executed Here)

```bash
python scripts/evaluate_finconrank_strict_nested.py --run-dir <new-complete-frozen-run>
python scripts/correct_finconrank_strict_nested_official_metrics.py \
  --run-dir <new-complete-frozen-run> --project-dir <this-repository>
```

Use only after all outer predictions/configs are frozen. Never rerun selection
after observing held-out metrics. Official corrected reporting is authoritative;
the correction must not be omitted. The included code contains the actual
report-group cluster bootstrap/sign-flip and separate Holm families; these methods
were not re-executed for packaging. No statistical results are supplied in this
archive. Missing/changed input manifests require independent provenance checking,
not removal of guards or silent reuse of old scores.
