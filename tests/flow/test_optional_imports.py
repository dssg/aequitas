import subprocess
import sys
import textwrap
import unittest

FLOW_DEPENDENCIES = {
    "fairgbm",
    "fairlearn",
    "hpt",
    "hydra",
    "lightgbm",
    "omegaconf",
    "optuna",
    "torch",
}


def run_without(modules: set[str], code: str) -> subprocess.CompletedProcess:
    """Run code in a new interpreter where `modules` cannot be imported, as if they
    were not installed."""
    block_imports = f"""
        import sys

        class BlockImports:
            def find_spec(self, name, path=None, target=None):
                if name.partition(".")[0] in {sorted(modules)!r}:
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

        sys.meta_path.insert(0, BlockImports())
    """
    script = textwrap.dedent(block_imports) + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )


class TestOptionalImports(unittest.TestCase):
    def test_datasets_without_flow_dependencies(self):
        result = run_without(
            FLOW_DEPENDENCIES,
            """
            from aequitas.flow.datasets import BankAccountFraud, FolkTables
            from aequitas.flow.datasets import GenericDataset
            """,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_flow_without_fairgbm(self):
        result = run_without(
            {"fairgbm"},
            """
            from aequitas.flow import DefaultExperiment, Experiment
            from aequitas.flow.methods.inprocessing import FairlearnClassifier
            from aequitas.flow.utils import ConfigReader

            try:
                from aequitas.flow.methods.inprocessing import FairGBM
            except ImportError:
                pass
            else:
                raise AssertionError("FairGBM was imported without fairgbm")
            """,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
