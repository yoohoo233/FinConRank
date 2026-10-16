from __future__ import annotations

import unittest

from run_major_revision_cost_benchmark import deduplicate, stable_key


class CostBenchmarkProtocolTest(unittest.TestCase):
    def test_deterministic_selection_key(self) -> None:
        self.assertEqual(stable_key("x"), stable_key("x"))
        self.assertNotEqual(stable_key("x"), stable_key("y"))

    def test_deduplicate_preserves_first_occurrence(self) -> None:
        rows = deduplicate([
            {"corpus_id": "a", "score": 1},
            {"corpus_id": "a", "score": 2},
            {"corpus_id": "b", "score": 3},
        ])
        self.assertEqual(rows, [{"corpus_id": "a", "score": 1}, {"corpus_id": "b", "score": 3}])


if __name__ == "__main__":
    unittest.main()
