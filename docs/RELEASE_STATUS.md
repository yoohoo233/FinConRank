# Release Status And Missing Inputs

## Verified During Packaging

Only static syntax/import-location/configuration checks, safe CLI help and
synthetic query-parser checks are authorized. Detailed results are in the separate
review archive, not in this public archive. All originally protocol-locked source
files selected for this release matched their recorded hashes before modification.
The public copies differ only in documented release-local path/import adaptations
and one descriptive provenance-status label. Algorithms, prompts, score direction,
hyperparameters, group assignments and statistical methods were not retuned.

## Organized, Not Executed

The scientific scripts are collected with their local dependencies. No training,
inference, dataset preprocessing, OCR, retrieval metric computation or statistical
analysis was executed for this release. A help/import check is not model validation.
The corrected official unique-document reporting script is included. The earlier
evaluation script remains a dependency/historical stage; do not use its pre-correction
metrics in place of corrected official reporting. Correction deduplicates documents
and restores official metric semantics without retraining or selecting new models.

## Required But Not Bundled

1. Original FinRAGBench-V report archives, page images, queries and qrels, obtained
   separately from the benchmark authors with their applicable conditions. This
   code archive does not grant financial-report image redistribution rights.
2. Qwen base and processor; original Mono initialization and FinConRank fold adapters.
3. ColQwen2 embeddings/candidate output. The local late-interaction scorer is
   included, but the official embedding-generation toolkit source tree was not
   copied: the available local official-code copy had no top-level license/README
   evidence. Obtain an appropriately licensed official release separately.
4. Source constraints, grouped folds, merged OCR/PDF page/block text and router
   source-score JSONL inputs. Schemas and producers are described in EXECUTION_ORDER.
5. Original frozen prediction/checkpoint/input manifests for exact result auditing.
   They are intentionally not bundled because they contain experiment/data paths
   or dataset-derived material. Code release is not a substitute for these inputs.
6. Suitable CUDA environment for GPU stages and Tesseract/language packs for OCR.

## Compatibility Boundaries

- The original code loads existing local resources; it is not an automatic
  downloader or turnkey benchmark runner.
- Path overrides change locations, not scientific settings. Empty variable values
  are treated as absent in the shared path helper; leave unused overrides unset.
- Historical report prose and legacy artifact field names are retained in several
  scripts for correspondence. They describe query/report-ID grouping, not full
  page/report exclusion. See README's explicit shared-corpus boundary.
- `.env.example` and paths.example.json do not inject values automatically.
- No notebooks, authentication services, human labels, author-private files,
  weights, benchmark data, Git history, logs or old export archives are included.
- No new AI labeling or human-audit UI is included in this method-code release.
