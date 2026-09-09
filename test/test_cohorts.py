import importlib.util
import json
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORTS_PATH = os.path.join(REPO_ROOT, "reproducibility", "cohorts.json")
FIGURE_UTIL_PATH = os.path.join(REPO_ROOT, "scripts", "figures", "util.py")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCohorts(unittest.TestCase):
    """Guard reproducibility/cohorts.json against the registries it documents.

    cohorts.json is not read at runtime yet. These tests keep it from drifting away from
    SYNAPSE_DICT and COHORT_DICT, so that making it the single source of truth later is a
    no-op change.
    """

    @classmethod
    def setUpClass(cls):
        with open(COHORTS_PATH) as f:
            cls.doc = json.load(f)
        cls.axes = cls.doc["axes"]
        cls.members = {
            axis: {name: set(cohort["cochleae"]) for name, cohort in cohorts.items()}
            for axis, cohorts in cls.axes.items()
        }

    def test_axes_are_disjoint(self):
        for axis, cohorts in self.members.items():
            seen = {}
            for name, cochleae in cohorts.items():
                for cochlea in cochleae:
                    self.assertNotIn(
                        cochlea, seen,
                        f"{cochlea} is in both {seen.get(cochlea)} and {name} on axis {axis}")
                    seen[cochlea] = name

    def test_required_keys(self):
        for axis, cohorts in self.axes.items():
            for name, cohort in cohorts.items():
                self.assertIn("label", cohort, f"{axis}/{name}")
                self.assertIn("description", cohort, f"{axis}/{name}")
                self.assertTrue(cohort["cochleae"], f"{axis}/{name} has no members")
                self.assertEqual(sorted(cohort["cochleae"]), cohort["cochleae"],
                                 f"{axis}/{name} members are not sorted")

    def test_undefined_holds_only_unclassified_cochleae(self):
        """A cochlea with a preparation protocol is not undefined, only absent from "group"."""
        with_protocol = set()
        for cochleae in self.members["protocol"].values():
            with_protocol |= cochleae
        for cochlea in sorted(self.members["group"]["undefined"]):
            self.assertNotIn(cochlea, with_protocol,
                             f"{cochlea} has a protocol, so it does not belong to undefined")

    def test_rerun_of_targets_exist(self):
        classified = set()
        for cohorts in self.members.values():
            for cochleae in cohorts.values():
                classified |= cochleae
        for rerun, original in self.doc["rerun_of"].items():
            self.assertIn(rerun, classified, f"{rerun} is not listed in any cohort")
            self.assertIn(original, classified, f"{original} is not listed in any cohort")

    def test_agrees_with_synapse_dict(self):
        from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
        protocols = self.members["protocol"]
        for cochlea, info in SYNAPSE_DICT.items():
            protocol = info.get("protocol")
            if protocol is None:
                continue
            self.assertIn(protocol, protocols, f"unknown protocol {protocol} for {cochlea}")
            self.assertIn(cochlea, protocols[protocol],
                          f"{cochlea} is missing from protocol {protocol}")

    def test_agrees_with_figure_cohort_dict(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("scripts/figures/util.py needs matplotlib")
        util = _load_module("figure_util", FIGURE_UTIL_PATH)
        groups = self.members["group"]
        for name in util.COHORT_DICT:
            self.assertIn(name, groups, f"cohort {name} of COHORT_DICT is not documented")
            self.assertEqual(set(util.cohort_cochleae(name)), groups[name],
                             f"members of cohort {name} differ")

    def test_covers_the_reproducibility_parameter_files(self):
        classified = set()
        for cohorts in self.members.values():
            for cochleae in cohorts.values():
                classified |= cochleae
        directories = ["block_extraction", "processing"]
        checked = 0
        for directory in directories:
            root = os.path.join(REPO_ROOT, "reproducibility", directory)
            for name in sorted(os.listdir(root)):
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(root, name)) as f:
                    data = json.load(f)
                for entry in (data if isinstance(data, list) else [data]):
                    checked += 1
                    self.assertIn(entry["dataset_name"], classified,
                                  f"{directory}/{name} names an undocumented cochlea")
        # Guard against the loop passing vacuously if a directory is renamed again.
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
