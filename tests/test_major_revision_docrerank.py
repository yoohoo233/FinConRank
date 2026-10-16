from __future__ import annotations

import unittest

import torch

from run_major_revision_docrerank import (
    PROMPT_TEMPLATE,
    comparison_plan,
    direct_rank,
    query_key,
    score_probability,
)


class DocReRankProtocolTest(unittest.TestCase):
    def test_official_prompt_is_exact_and_query_is_inserted(self) -> None:
        self.assertEqual(
            PROMPT_TEMPLATE.format(query="Q"),
            "Assert the relevance of the previous image document to the following query, answer True or False. The query is: Q",
        )

    def test_true_probability_uses_only_true_false_logits(self) -> None:
        logits = torch.tensor([[0.0, 2.0, -2.0, 100.0]])
        score = score_probability(logits, 1, 2)
        self.assertAlmostEqual(float(score[0]), float(torch.softmax(torch.tensor([2.0, -2.0]), dim=0)[0]))

    def test_direct_rank_is_descending_with_colq_tie_break(self) -> None:
        candidates = [
            {"corpus_id": "b", "colqwen2_rank": 2, "docrerank_score": 0.5},
            {"corpus_id": "a", "colqwen2_rank": 1, "docrerank_score": 0.5},
            {"corpus_id": "c", "colqwen2_rank": 3, "docrerank_score": 0.9},
        ]
        ranked = direct_rank(candidates, "docrerank_score", "test")
        self.assertEqual([row["corpus_id"] for row in ranked], ["c", "a", "b"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])

    def test_direct_rank_deduplicates_before_score_sorting(self) -> None:
        candidates = [
            {"corpus_id": "a", "colqwen2_rank": 1, "docrerank_score": 0.2},
            {"corpus_id": "a", "colqwen2_rank": 2, "docrerank_score": 0.9},
            {"corpus_id": "b", "colqwen2_rank": 3, "docrerank_score": 0.5},
        ]
        ranked = direct_rank(candidates, "docrerank_score", "test")
        self.assertEqual([row["corpus_id"] for row in ranked], ["b", "a"])
        self.assertEqual([row["docrerank_score"] for row in ranked], [0.5, 0.2])

    def test_query_key_includes_language(self) -> None:
        self.assertNotEqual(
            query_key({"language": "en", "query_id": "same"}),
            query_key({"language": "zh", "query_id": "same"}),
        )

    def test_comparison_plan_is_fully_explicit_and_stable(self) -> None:
        plan = comparison_plan()
        self.assertEqual(len(plan), 8)
        self.assertEqual([row["comparison_index"] for row in plan], list(range(8)))
        self.assertEqual([row["seed"] for row in plan], [42 + index * 31 for index in range(8)])
        self.assertEqual(
            {(row["baseline"], row["language"], row["metric"]) for row in plan},
            {
                (baseline, language, metric)
                for baseline in ("monoqwen2vl", "finconrank")
                for language in ("en", "zh")
                for metric in ("nDCG@10", "MRR@10")
            },
        )


if __name__ == "__main__":
    unittest.main()
