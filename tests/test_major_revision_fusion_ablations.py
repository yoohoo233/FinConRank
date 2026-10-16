from __future__ import annotations

import unittest

import numpy as np

from run_major_revision_fusion_ablations import deduplicate_candidates, fuse_matched, simplex_grid


class FusionGridTest(unittest.TestCase):
    def test_candidate_deduplication_keeps_first_ranked_occurrence(self) -> None:
        rows = deduplicate_candidates([
            {"corpus_id": "a", "rank": 1, "score": 2.0},
            {"corpus_id": "a", "rank": 2, "score": 1.0},
            {"corpus_id": "b", "rank": 3, "score": 0.0},
        ])
        self.assertEqual([row["corpus_id"] for row in rows], ["a", "b"])
        self.assertEqual([row["rank"] for row in rows], [1, 2])

    def test_three_score_grid_size(self) -> None:
        grid = simplex_grid(3)
        self.assertEqual(len(grid), 21)
        self.assertTrue(all(np.isclose(weight.sum(), 1.0) for weight in grid))

    def test_zero_allowed_four_score_grid_contains_full_three_score_subspace(self) -> None:
        three = {tuple(weight.tolist()) for weight in simplex_grid(3)}
        four = simplex_grid(4)
        projected = {
            (float(weight[0]), float(weight[1]), float(weight[3]))
            for weight in four
            if np.isclose(weight[2], 0.0)
        }
        self.assertEqual(len(four), 56)
        self.assertEqual(projected, three)

    def test_positive_four_score_grid_matches_historical_budget(self) -> None:
        self.assertEqual(len(simplex_grid(4, require_fincon_positive=True)), 35)

    def test_zero_fincon_weight_reproduces_three_score_ranking(self) -> None:
        candidates = [
            {"corpus_id": "a", "rank": 1, "score": 3.0, "rerank_score": 0.2, "finconrank_score": -9.0},
            {"corpus_id": "b", "rank": 2, "score": 2.0, "rerank_score": 0.8, "finconrank_score": 9.0},
        ]
        metric = {"a": 0.7, "b": 0.1}
        shared = {
            "colq": {"mean": 2.5, "std": 0.5},
            "mono": {"mean": 0.5, "std": 0.3},
            "metric": {"mean": 0.4, "std": 0.3},
        }
        four = {**shared, "fincon": {"mean": 0.0, "std": 9.0}}
        three_weights = np.asarray([0.2, 0.4, 0.4])
        four_weights = np.asarray([0.2, 0.4, 0.0, 0.4])
        three = fuse_matched(candidates, metric, shared, three_weights, False, "three")
        zero = fuse_matched(candidates, metric, four, four_weights, True, "zero")
        self.assertEqual([row["corpus_id"] for row in three], [row["corpus_id"] for row in zero])
        self.assertTrue(np.allclose([row["fusion_score"] for row in three], [row["fusion_score"] for row in zero]))


if __name__ == "__main__":
    unittest.main()
