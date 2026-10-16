from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from major_revision_gpu_scheduler import (
    GPUStatus,
    MAX_PHYSICAL_GPUS,
    acquire,
    descendants,
    eligible,
    release,
    select_stable_gpus,
)


def gpu(index: int, *, used: int = 10, util: int = 0, pids: tuple[int, ...] = (), known: bool = True) -> GPUStatus:
    return GPUStatus(
        index=index, uuid=f"GPU-{index}", name="test", memory_total_mib=24564,
        memory_used_mib=used, memory_free_mib=24564-used,
        utilization_percent=util, compute_pids=pids, process_visibility_known=known,
    )


class SchedulerDecisionTest(unittest.TestCase):
    def test_idle_card_counts_and_hard_cap(self) -> None:
        for count in (0, 1, 3, 4, 8):
            snapshot = [gpu(index, used=10 if index < count else 5000) for index in range(8)]
            selected = select_stable_gpus([snapshot, snapshot, snapshot], set(), 20000, requested=8)
            self.assertEqual(len(selected), min(count, 4, 6))
            self.assertLessEqual(len({item.uuid for item in selected}), MAX_PHYSICAL_GPUS)
            self.assertTrue(all(item.index < 6 for item in selected))

    def test_external_goal_and_unknown_process_states_are_rejected(self) -> None:
        self.assertEqual(eligible(gpu(0, pids=(99,)), set(), 20000)[1], "external_compute_process")
        self.assertEqual(eligible(gpu(0, pids=(99,)), {99}, 20000)[1], "already_goal_leased")
        self.assertEqual(eligible(gpu(0, known=False), set(), 20000)[1], "process_visibility_unknown")
        self.assertEqual(eligible(gpu(6), set(), 20000)[1], "project_not_authorized")

    def test_three_consecutive_observations_are_required(self) -> None:
        self.assertEqual(select_stable_gpus([[gpu(0)], [gpu(0)]], set(), 20000), [])
        snapshots = [[gpu(0)], [gpu(0, util=6)], [gpu(0)]]
        self.assertEqual(select_stable_gpus(snapshots, set(), 20000), [])

    def test_child_escape_detection(self) -> None:
        tree = {11: 10, 12: 11, 20: 1}
        self.assertEqual(descendants(10, tree), {10, 11, 12})

    def test_lease_failure_cleanup_and_resume_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leases.json"
            acquire(path, "run", "task-a", 123, gpu(0))
            with self.assertRaises(RuntimeError):
                acquire(path, "run", "task-b", 124, gpu(0))
            release(path, "task-a", "failed")
            acquire(path, "run", "task-b", 124, gpu(0))
            state = json.loads(path.read_text())
            self.assertEqual([item["status"] for item in state["leases"]], ["failed", "active"])

    def test_global_lease_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "leases.json"
            for index in range(4):
                acquire(path, "run", f"task-{index}", 100 + index, gpu(index))
            with self.assertRaises(RuntimeError):
                acquire(path, "run", "task-4", 104, gpu(4))


if __name__ == "__main__":
    unittest.main()
