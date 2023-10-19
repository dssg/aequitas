# flake8: noqa

from pathlib import Path
from typing import Union

from omegaconf import DictConfig

from .experiment import Experiment

lightgbm_config = {
    "lightgbm": {
        "classpath": "aequitas.fairflow.methods.inprocessing.lightgbm.LightGBM",
        "args": {
            "boosting_type": ["dart", "gbdt"],
            "enable_bundle": [False],
            "n_estimators": {"type": "int", "range": [100, 1000]},
            "num_leaves": {"type": "int", "range": [10, 1000]},
            "min_child_samples": {
                "type": "int",
                "range": [1, 500],
                "log": True,
            },
            "learning_rate": {"type": "float", "range": [0.001, 0.1]},
            "n_jobs": [1],
        },
    },
}

default_methods = [
    {
        "undersampling": {
            "preprocessing": {
                "undersampling": {
                    "classpath": "aequitas.fairflow.methods.preprocessing.prevalence_sample.PrevalenceSampling",
                    "args": {
                        "alpha": {"type": "float", "range": [0.5, 1]},
                    },
                },
            },
            "inprocessing": lightgbm_config,
        },
    },
    {
        "oversampling": {
            "preprocessing": {
                "oversampling": {
                    "classpath": "aequitas.fairflow.methods.preprocessing.prevalence_sample.PrevalenceSampling",
                    "args": {
                        "alpha": {"type": "float", "range": [0.5, 1]},
                        "strategy": ["oversample"],
                    },
                },
            },
            "inprocessing": lightgbm_config,
        },
    },
    {
        "lightgbm_baseline": {
            "inprocessing": lightgbm_config,
        },
    },
    {
        "fairgbm_punitive": {
            "inprocessing": {
                "fairgbm": {
                    "classpath": "aequitas.fairflow.methods.inprocessing.fairgbm.FairGBM",
                    "args": {
                        "boosting_type": ["dart", "gbdt"],
                        "enable_bundle": [False],
                        "n_estimators": {"type": "int", "range": [100, 1000]},
                        "num_leaves": {"type": "int", "range": [10, 1000]},
                        "min_child_samples": {
                            "type": "int",
                            "range": [1, 500],
                            "log": True,
                        },
                        "learning_rate": {"type": "float", "range": [0.001, 0.1]},
                        "n_jobs": [1],
                        "constraint_stepwise_proxy": ["cross_entropy"],
                        "multiplier_learning_rate": {
                            "type": "float",
                            "range": [0.01, 0.1],
                            "log": True,
                        },
                        "constraint_type": "fpr",
                    },
                },
            },
        },
    },
    {
        "fairgbm_assistive": {
            "inprocessing": {
                "fairgbm": {
                    "classpath": "aequitas.fairflow.methods.inprocessing.fairgbm.FairGBM",
                    "args": {
                        "boosting_type": ["dart", "gbdt"],
                        "enable_bundle": [False],
                        "n_estimators": {"type": "int", "range": [100, 1000]},
                        "num_leaves": {"type": "int", "range": [10, 1000]},
                        "min_child_samples": {
                            "type": "int",
                            "range": [1, 500],
                            "log": True,
                        },
                        "learning_rate": {"type": "float", "range": [0.001, 0.1]},
                        "n_jobs": [1],
                        "constraint_stepwise_proxy": ["cross_entropy"],
                        "multiplier_learning_rate": {
                            "type": "float",
                            "range": [0.01, 0.1],
                            "log": True,
                        },
                        "constraint_type": "fnr",
                    },
                },
            },
        },
    },
]
# For now, abstaining from other configurations as they are highly dependant on the fairness metric and context.
# Because of this we are not including thresholding.


# Experiment class with default configurations for methods
class DefaultExperiment(Experiment):
    def __init__(
        self,
        train_path: Union[str, Path],
        validation_path: Union[str, Path],
        test_path: Union[str, Path],
        categorical_columns: list[str],
        target_column: str,
        extension: str = "parquet",
    ):
        dataset_config = {
            "dataset": {
                "classpath": "aequitas.fairflow.datasets.generic.GenericDataset",
                "args": {
                    "train_path": train_path,
                    "validation_path": validation_path,
                    "test_path": test_path,
                    "categorical_columns": categorical_columns,
                    "target_column": target_column,
                    "extension": extension,
                },
            }
        }

        config = {
            "methods": default_methods,
            "datasets": dataset_config,
            "optimization": {
                "n_trials": 100,
                "n_jobs": 1,
                "sampler": "RandomSampler",
                "sampler_args": {"seed": 42},
            },
        }
        super().__init__(config=DictConfig(config))
