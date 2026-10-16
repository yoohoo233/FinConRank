# FinConRank Research Code

## Revision Release Additions

This candidate extends the original public-code archive without replacing its
historical main-experiment algorithms. Author-provided public code reference:
`229d81809d51fe887b3538ec1dfcd0a97f8a8e14` (not fetched or verified remotely).
The original main `fusion_grid` still requires a positive FinConRank coefficient.
Separate revision entry points implement three-score fusion and a four-score
search allowing a zero FinConRank coefficient; the historical grid was not edited.

See [Revision Entry Points](docs/REVISION_ENTRY_POINTS.md) for fusion ablations,
same-ColQwen2-top-20 DocReRank comparison, descriptive local training sensitivity,
paired statistical reporting, parser diagnostics and measured costs. Source Data
is a separate archive of already frozen outputs, not regenerated experiments.
The T2 files are archival runners with explicit prerequisites, not a new turnkey
training framework or a page-isolation repair. No same-budget training comparison
or new environment end-to-end verification is claimed.

Code-only preparation for public release under **Apache-2.0**, approved by the
author and coauthors. This archive has not been committed or published to GitHub.
Update the existing public repository from a clean release copy; do not import
private Git history, data, credentials or local experiment artifacts.

## Scope And Evaluation Boundary

FinConRank continues a MonoQwen2-VL LoRA using financial hard-negative pairs and
query-constraint prompts. The score is the final-position `True` minus `False`
logit. The collection also includes original query grouping, inner cross-fitted
selection, Metric-Aware calibration, score fusion and clustered statistics code.

**Evaluation uses query grouping on a shared fixed document corpus.** Query
groups use report identifiers, but negative pages can cross report boundaries.
Query/group separation is not document-page isolation. Do not describe this as
strict unseen-page or unseen-report generalization. Historical filenames containing
`strict_nested` are retained for traceability, not as a broader isolation claim.
Human quality-audit labels are not training features or model-selection inputs.

This is not a complete, independently executed end-to-end reproduction. No model
training, scoring, metrics or statistical experiments were run to prepare this
package. See [Release Status](docs/RELEASE_STATUS.md) for missing inputs and checks.

## Model Collection

- Model repository: https://huggingface.co/yoohoo233/FinConRank
- Fixed model revision: `cf814c526580a8ddc011b9cd12429765580777ff`
- English: `en/outer_0` ... `en/outer_4`; Chinese: `zh/outer_0` ... `zh/outer_4`.
- These are ten fold-specific adapters, not one all-data deployment model. Do not
  choose a fold by held-out performance or average adapters.
- Load `Qwen/Qwen2-VL-2B-Instruct` followed by **one FinConRank adapter directly**.
  Training started from `lightonai/MonoQwen2-VL-v0.1`; do not stack Mono again.
- Pinned original Qwen revision: `895c3a49bc3fa70a340399125c650a463535e71c`.
- Pinned original Mono revision: `5d718b52d534290938ed79deedc620a7f6a6254f`.

The repository/revision above is author-provided metadata, not a remote verification
performed during packaging. Weights and data are not in this archive. See NOTICE
for upstream attribution and license-evidence limitations.

## Layout

| Directory | Purpose |
| --- | --- |
| `src/finconrank/` | Original parser/dictionaries, common helpers, grouped protocol helpers; release-only path resolver |
| `scripts/` | Preprocessing, negative construction, training, inference, grouping, selection, fusion, evaluation and statistics |
| `configs/finconrank/` | Protocol configuration copied from the frozen experiment |
| `configs/paths.example.json` | Input-location schema; documentation only, not an automatic loader |
| `tests/` | Small synthetic parser and configuration checks, without real data |
| `docs/` | Execution order, schemas, correspondence, limitations and validation scope |

## Environment And Paths

`requirements.txt` records installed dependency versions observed during code
packaging; it is not a newly verified historical execution environment or a claim
that installing on another CUDA stack will succeed. No dependency was installed
or upgraded for this package. OCR additionally needs an external Tesseract binary
and language packs. Reuse a suitable environment before attempting data/GPU stages.

Each script bootstraps only this release's `src/` directory. No original server
PYTHONPATH is needed. The `scripts` directory remains the location for sibling
script imports. You can optionally install the local package in your own environment.
`.env.example` contains only empty variable names; it is not loaded automatically.
Set the desired variables explicitly. Never commit credentials or local data.

```bash
export FINCONRANK_BASE_MODEL="<local-pinned-Qwen-snapshot>"
export FINCONRANK_INITIAL_ADAPTER="<local-pinned-Mono-adapter>"
export FINRAGBENCH_DATA_ROOT="<read-only-original-benchmark>"
export FINCONRANK_EN_IMAGES="<English-page-images>"
export FINCONRANK_ZH_IMAGES="<Chinese-page-images>"
export FINCONRANK_BASELINE_DIR="<tab-safe-baseline-directory>"
export FINCONRANK_SOURCE_RUN="<source-run-with-constraints-and-Mono-scores>"
export FINCONRANK_SOURCE_OOF="<group-fold-manifest-directory>"
export FINCONRANK_INDEX_DIR="<merged-OCR-PDF-evidence-directory>"
export FINCONRANK_ROUTER_DIR="<router-source-score-directory>"
export FINCONRANK_CONSTRAINTS="<query_constraints.jsonl>"
```

Defaults preserve historical relative layout; the dated folder names are expected
input locations, not bundled data. Set explicit overrides or recreate the layout.
All derived outputs belong outside the read-only original benchmark. Do not use
any user-created `prepared` directory as the original dataset.

## Usage

See [Execution Order](docs/EXECUTION_ORDER.md) for stage-by-stage templates and
required external inputs. Merely running `--help` does not validate a scientific
stage. Do not run evaluation on a frozen run: use a fresh output directory, respect
the prediction freeze guards, and choose parameters only from allowed inner folds.

Safe example:

```bash
python scripts/train_finconrank.py --help
python tests/test_release_safety.py
```

No experimental result tables are bundled or reconstructed here. Full fusion
performance is not a single adapter's metric. Weight publication alone does not
establish complete reproduction or stronger generalization claims.
