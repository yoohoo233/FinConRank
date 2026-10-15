"""Synthetic parser and configuration tests only; no relevance metrics or models."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finconrank.constraints import extract_constraints
from finconrank.release_paths import release_path


class ReleaseSafetyTest(unittest.TestCase):
    def test_english_constraints(self):
        result = extract_constraints('What was revenue in 2023 in USD million?')
        self.assertIn('revenue', result['canonical_metrics'])
        self.assertEqual(result['year_terms'], ['2023'])
        self.assertIn('usd', result['unit_terms'])

    def test_chinese_constraints(self):
        result = extract_constraints('2022年净利润相比2021年增长多少？')
        self.assertIn('net_income', result['canonical_metrics'])
        self.assertEqual(result['year_terms'], ['2021', '2022'])
        self.assertIn('增长', result['operation_terms'])

    def test_unknown_remains_unknown(self):
        result = extract_constraints('Explain this.')
        self.assertIsNone(result['entity'])
        self.assertEqual(result['canonical_metrics'], [])

    def test_path_default_and_override(self):
        with patch.dict(os.environ, {'FINCONRANK_BASE_MODEL': ''}):
            self.assertEqual(Path(release_path('FINCONRANK_BASE_MODEL', 'models/example')), ROOT / 'models/example')
        with patch.dict(os.environ, {'FINCONRANK_BASE_MODEL': 'local-models/example'}):
            self.assertEqual(Path(release_path('FINCONRANK_BASE_MODEL', 'models/example')), ROOT / 'local-models/example')

    def test_ten_fold_structure(self):
        config = json.loads((ROOT / 'configs/paths.example.json').read_text())
        self.assertEqual(len(config['adapter_subfolders']['en']), 5)
        self.assertEqual(len(config['adapter_subfolders']['zh']), 5)
        self.assertEqual(config['model_revision'], 'cf814c526580a8ddc011b9cd12429765580777ff')


if __name__ == '__main__':
    unittest.main()
