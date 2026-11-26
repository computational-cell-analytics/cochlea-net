import ast
import importlib
import inspect
import os
import unittest
from pathlib import Path
from unittest import mock

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "flamingo_tools"


def _package_modules():
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        yield path.relative_to(PACKAGE_DIR.parent), ast.parse(path.read_text(), filename=str(path))


def _elf_parallel_names(tree):
    """Map the local names of elf.parallel functions and modules in a module to their objects."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("elf.parallel"):
                    names[alias.asname or alias.name] = importlib.import_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "elf":
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                source = obj.__name__ if inspect.ismodule(obj) else getattr(obj, "__module__", "")
                if source.startswith("elf.parallel"):
                    names[alias.asname or alias.name] = obj
    return names


def _sets_n_threads(call, function):
    parameters = list(inspect.signature(function).parameters)
    if "n_threads" not in parameters:
        return True
    if any(keyword.arg in ("n_threads", None) for keyword in call.keywords):
        return True
    return parameters.index("n_threads") < len(call.args)


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


# A thread count from the core count of the node oversubscribes a job on a shared Slurm partition.
class TestThreadCountsFollowTheAllocation(unittest.TestCase):
    def test_no_cpu_count_outside_cpu_utils(self):
        offenders = []
        for path, tree in _package_modules():
            if path.name == "cpu_utils.py":
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                    if name in ("cpu_count", "process_cpu_count"):
                        offenders.append(f"{path}:{node.lineno}")
        self.assertEqual(offenders, [], "Use flamingo_tools.cpu_utils.available_cpus instead.")

    def test_elf_parallel_calls_set_n_threads(self):
        # elf.parallel uses multiprocessing.cpu_count() when n_threads is None.
        offenders = []
        for path, tree in _package_modules():
            names = _elf_parallel_names(tree)
            if not names:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name):
                    function = names.get(func.id)
                elif isinstance(func, ast.Attribute) and ast.unparse(func.value) in names:
                    function = getattr(names[ast.unparse(func.value)], func.attr, None)
                else:
                    function = None
                if callable(function) and not inspect.ismodule(function) and not _sets_n_threads(node, function):
                    offenders.append(f"{path}:{node.lineno} {ast.unparse(func)}")
        self.assertEqual(offenders, [], "Pass n_threads from flamingo_tools.cpu_utils.available_cpus.")


if __name__ == "__main__":
    unittest.main()
