# Source Correspondence

The selection is based on the frozen nested protocol and its documented workflow, not file modification times.
Initial protocol-locked sources match their frozen SHA-256 before release-only edits. Helpers and later official-metric correction files not in that initial lock are identified separately; no historical-version claim is inferred for them.
Release-only edits: location resolution, script-local import bootstrap, blank environment fallback handling, and descriptive provenance status. No new training method was implemented.

| File | Source SHA-256 | Initial protocol match |
| --- | --- | --- |
| `scripts/build_finconrank_training_data.py` | `066fc68761d041373128b10a16ff83fac7f7b1199bb2bb771b309d5e2100c6e9` | True |
| `scripts/build_real_multigranularity_index.py` | `eacdd0c5d134d46a11ae15036c2620d98a4c7b214e2521eafa32415a367ec6c9` | not_in_initial_lock |
| `scripts/build_tesseract_ocr_index.py` | `011f833dc2bb753af3de9b85d1e40908dfb0508941f2e690f3cd74ce9887eea5` | not_in_initial_lock |
| `scripts/correct_finconrank_strict_nested_official_metrics.py` | `47219a9de1c561c510fd9b8919cb88e8925aae1cad2ae53c10de29f7901da05d` | not_in_initial_lock |
| `scripts/evaluate_and_export_retrieval.py` | `fbc1470693b3fffa1ea791a798f88c1f7819da975cdd6cf8dc2984d7e0ea0523` | not_in_initial_lock |
| `scripts/evaluate_finconrank.py` | `6399775f7f8945032e589f3ba8d02b8c10571911c1780ba28a0b80796e391704` | True |
| `scripts/evaluate_finconrank_strict_nested.py` | `1d8f811018af436ce82cb019c52d39224cd80385bc29c4b12560079c92107143` | True |
| `scripts/finconrank_strict_nested_scheduler.py` | `9e68a3d6d5caa7ed4a9b7676289839b33d2611193fe29029d22947e0acb500c8` | True |
| `scripts/finconrank_strict_nested_scheduler_limited.py` | `98310af37505389dcdaab9d8bcd04c6490b2cd10cf5310b62843db3c2dcd4749` | not_in_initial_lock |
| `scripts/infer_finconrank.py` | `bf3142348c8d39b294b96d25c985b6b0601353d5572c7f4e85b84dc80de0091d` | not_in_initial_lock |
| `scripts/merge_ocr_pdf_multigranularity_index.py` | `297bb7ee353ee24a927f4e153d0b935d4ff17dbe15dd59fd264540863e74373e` | not_in_initial_lock |
| `scripts/prepare_finconrank_oof.py` | `7a80e91bd9dd6e0cc5aced5a0b63361ae42b4d1ccabee780d5d1d5174375c8a6` | not_in_initial_lock |
| `scripts/prepare_finconrank_strict_nested.py` | `e2070d86bfca36696626aa0acfd9b411f3bbc47d2d645b86910c0502f2d676ff` | True |
| `scripts/prepare_official_finragbench_inputs.py` | `f17dee609eedb5a727c92eacbf58dab7f6ea38a927df78e768eea3a0e9e5259b` | not_in_initial_lock |
| `scripts/reconstruct_tab_safe_trec.py` | `c965b646504dff2e6293c1b314acbe3f165183b14c5517e3b3ae24a8049dff5c` | not_in_initial_lock |
| `scripts/rerank_with_monoqwen2vl.py` | `2ac562f3cfb5f1554d50495002be1ccce3267c23a8be192b03a9ec1a11e6db5d` | not_in_initial_lock |
| `scripts/run_financial_metric_aware_reranking.py` | `259c35c39c6ca985b04ab42a927a467ff892e9de0ed4b61fa65ca1f914111a08` | True |
| `scripts/run_finconrank_audit.py` | `06be5b46c1969a4d11909f0fc98c3a9505c27c6a4ed8d64d51d4a7d150aa26a7` | not_in_initial_lock |
| `scripts/run_finconrank_strict_nested_task.py` | `b2e9df182869b20725154167739281f1679700d2de5da801a2b475603beed763` | True |
| `scripts/run_real_router_multigranularity.py` | `32aa6bac4469302a0700436647fee51c878da1a27fcde204d62fd6c37a289e46` | not_in_initial_lock |
| `scripts/score_colqwen2_embeddings.py` | `17ce1a0b1d12b0d0edb3ee78940475e82a77c59fa88a070b7b00815b9da1bc41` | not_in_initial_lock |
| `scripts/select_finconrank_strict_nested.py` | `709919413e245401f4678d0fcedc6a26fc191a612bcd71c57861971f2907a4ba` | True |
| `scripts/train_finconrank.py` | `34884f40195d946d8032e40dc2499c73d9868538a798bd6ee7fff23acceebe8b` | True |
| `src/finconrank/__init__.py` | `0d9d40730c70f7c3c06e3d466702b8229aa7e5cdb864fe4e135d8e6b90f95fd4` | not_in_initial_lock |
| `src/finconrank/common.py` | `a2cd4d1ec9ded1f5fc5a07f7087fdf1a08381ef69a5744fbc1f7a286bcb43637` | not_in_initial_lock |
| `src/finconrank/constraints.py` | `508085f045482b651171b284e25eb7e6760636d3b5cfcfd2935abbee635c30aa` | not_in_initial_lock |
| `src/finconrank/strict_nested.py` | `b64834517a52907542a658b7466fbcee64ab01a8c6431eaffa8e3b24bf1215af` | True |
| `configs/finconrank/strict_nested_group_oof.yaml` | `aec371fb3cb82d00bdd5cc418579776bab666d2412ab4a916b42f8f65aa3d2c2` | True |
