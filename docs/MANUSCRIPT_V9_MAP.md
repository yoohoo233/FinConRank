# Final v9 Revision Entry Map

| Revision entry | Manuscript location | Source Data |
| --- | --- | --- |
| run_major_revision_fusion_ablations.py | Section 4.3, Table 4, Supplementary S2/S4 | 02_FUSION |
| run_major_revision_docrerank.py | Section 4.4, Table 5, Supplementary S5 | 03_DOCReRank |
| experiments/t2/06_REPRODUCIBILITY/scripts/ | Section 4.7, Supplementary S5/Table S7 | 04_T2 |
| run_major_revision_cached_diagnostics.py | Section 3.2.2/5.4, Supplementary S1/S3 | 05_DIAGNOSTICS |
| run_major_revision_cost_benchmark.py and cost helpers | Section 3.3.3/5.4, Supplementary S4/Table S6 | 06_COST |
| Original matched-negative analysis | Section 4.5, Table 6 | 09_MATCHED_NEGATIVES |

The original select_finconrank_strict_nested.py preserves its positive-FinConRank-weight grid. The revision fusion entry independently implements three-score and zero-allowed grids; it does not silently replace the historical result.

These are published source/archival entry points, not a freshly verified turnkey environment. External data, models, original manifests and validated relocated locks remain prerequisites. In particular, do not execute detached T2 launchers during review.
