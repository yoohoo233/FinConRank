
# Release-local import bootstrap; no original project path is used.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "src"))
_release_sys.path.insert(0, str(_ReleasePath(__file__).resolve().parents[4] / "scripts"))
import unittest
from pathlib import Path
from t2_common import ROOT, read


class PlanTest(unittest.TestCase):
    def test_matrix_counts(self):
        plan = read(ROOT / "input/FinConRank_T2_fixed_corpus_execution_plan_20261013_v2.json")
        self.assertEqual(len(plan["units"])*len(plan["training_families"]),20)
        self.assertEqual(len(plan["units"])*len(plan["evaluated_conditions"]),28)
        self.assertEqual(sum(len(f["score_epochs"]) for f in plan["training_families"]),7)

    def test_reference_and_fixed_seeds(self):
        plan = read(ROOT / "input/FinConRank_T2_fixed_corpus_execution_plan_20261013_v2.json")
        self.assertEqual([u["training_seed"] for u in plan["units"]],[42,43,10042,10043])
        reference=plan["training_families"][0]
        self.assertEqual(reference["train_epochs"],4)
        self.assertEqual(reference["score_epochs"],[2,3,4])
        self.assertEqual(plan["limits"]["maximum_physical_gpus"],4)

    def test_no_outer_label_read_in_workers(self):
        folder=Path(__file__).parent
        for name in ("score_v2.py","scheduler_v2.py"):
            self.assertNotIn("sealed_qrels",(folder/name).read_text())

    def test_idle_gpu_rules(self):
        from major_revision_gpu_scheduler import GPUStatus
        from scheduler_v2 import free
        def gpu(index,pids=(),used=10,util=0,known=True):
            return GPUStatus(index,f"GPU-{index}","test",24000,used,24000-used,util,pids,known)
        self.assertTrue(free(gpu(0)))
        self.assertFalse(free(gpu(0,pids=(123,))))
        self.assertFalse(free(gpu(0,known=False)))
        self.assertFalse(free(gpu(0,used=1500)))
        self.assertFalse(free(gpu(0,util=5)))
        self.assertFalse(free(gpu(6)))
        self.assertFalse(free(gpu(7)))


if __name__ == "__main__":
    unittest.main()
