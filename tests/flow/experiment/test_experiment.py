import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aequitas.flow.experiment.default import DefaultExperiment
from aequitas.flow.experiment.experiment import Experiment
from aequitas.flow.methods.postprocessing.threshold import Threshold


def make_dataframe(n_rows=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n_rows), "x2": rng.normal(size=n_rows)})
    s = rng.integers(0, 2, size=n_rows)
    df["y"] = ((df["x1"] + 0.5 * s + rng.normal(size=n_rows)) > 0.5).astype(int)
    df["s"] = pd.Categorical(s)
    return df


def make_config():
    return {
        "methods": [
            {
                "small_lightgbm": {
                    "inprocessing": {
                        "lightgbm": {
                            "classpath": "aequitas.flow.methods.base_estimator.LightGBM",
                            "args": {"n_estimators": [5], "num_leaves": [4]},
                        }
                    }
                }
            }
        ],
        "datasets": [
            {
                "Dataset": {
                    "classpath": "aequitas.flow.datasets.GenericDataset",
                    "threshold": {"threshold_type": "fixed", "threshold_value": 0.5},
                    "args": {
                        "df": make_dataframe(),
                        "label_column": "y",
                        "sensitive_column": "s",
                    },
                }
            }
        ],
        "optimization": {
            "n_trials": 1,
            "n_jobs": 1,
            "sampler": "RandomSampler",
            "sampler_args": {"seed": 42},
        },
    }


class TestDefaultExperiment(unittest.TestCase):
    def test_methods_are_not_repeated(self):
        for methods in ("all", ["preprocessing"], ["inprocessing"]):
            config = DefaultExperiment._generate_config(
                dataset_config={},
                methods=methods,
                experiment_size="test",
                use_baseline=True,
            )
            names = [name for method in config["methods"] for name in method]
            self.assertEqual(len(names), len(set(names)), methods)

    def test_threshold_config(self):
        experiment = DefaultExperiment(
            make_dataframe(),
            label_column="y",
            sensitive_column="s",
            threshold_type="fpr",
            score_threshold=0.05,
        )
        threshold = Threshold(**experiment.config.datasets[0]["Dataset"].threshold)
        self.assertEqual(threshold.threshold_type, "fpr")
        self.assertEqual(threshold.threshold_value, 0.05)


class TestExperiment(unittest.TestCase):
    def test_run(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Experiment(config=make_config(), save_folder=Path(folder))
            experiment.run()
            self.assertEqual(len(list(Path(folder).rglob("results.pickle"))), 1)

    def test_run_without_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            experiment = Experiment(
                config=make_config(), save_artifacts=False, save_folder=Path(folder)
            )
            experiment.run()
            self.assertEqual(list(Path(folder).iterdir()), [])
