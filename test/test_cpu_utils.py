import os
import unittest
from unittest import mock


class TestAvailableCpus(unittest.TestCase):
    def test_slurm_allocation_comes_first(self):
        from flamingo_tools.cpu_utils import available_cpus

        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "3"}):
            self.assertEqual(available_cpus(), 3)

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "needs os.sched_getaffinity")
    def test_affinity_mask_without_slurm(self):
        from flamingo_tools.cpu_utils import available_cpus

        env = {key: value for key, value in os.environ.items() if key != "SLURM_CPUS_PER_TASK"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(available_cpus(), len(os.sched_getaffinity(0)))


if __name__ == "__main__":
    unittest.main()
