from __future__ import annotations

import unittest

from run_major_revision_cached_diagnostics import field_present, precision_at, unique_ids


class CachedDiagnosticsTest(unittest.TestCase):
    def test_precision_uses_fixed_k_denominator_and_binary_relevance(self) -> None:
        candidates = [{"corpus_id": "a", "rank": 1}, {"corpus_id": "b", "rank": 2}]
        self.assertEqual(precision_at(candidates, {"a": 2, "b": 0}, 5), 0.2)

    def test_candidates_are_deduplicated_by_first_rank(self) -> None:
        candidates = [
            {"corpus_id": "a", "rank": 2},
            {"corpus_id": "a", "rank": 1},
            {"corpus_id": "b", "rank": 3},
        ]
        self.assertEqual(unique_ids(candidates), ["a", "b"])

    def test_unknown_statement_is_not_present(self) -> None:
        self.assertFalse(field_present({"statement_type": "unknown"}, "statement_type"))
        self.assertTrue(field_present({"statement_type": "cash_flow"}, "statement_type"))
        self.assertFalse(field_present({"year_terms": []}, "year_terms"))
        self.assertTrue(field_present({"year_terms": ["2023"]}, "year_terms"))


if __name__ == "__main__":
    unittest.main()
