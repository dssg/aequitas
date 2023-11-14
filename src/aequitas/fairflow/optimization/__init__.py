import dataclasses
import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from hpt.suggest import suggest_callable_hyperparams
from omegaconf import DictConfig, OmegaConf
from optuna.trial import BaseTrial

from ..evaluation import evaluate_fairness, evaluate_performance
from ..methods import InProcessing, PostProcessing, PreProcessing
from ..methods.postprocessing.threshold import Threshold


@dataclasses.dataclass
class Result:
    id: int
    hyperparameters: dict
    validation_results: dict
    test_results: dict = None
    train_results: dict = None
    fit_time: float = None
    validation_time: float = None
    test_time: float = None
    algorithm: str = None


class ObjectiveFunction:
    """Callable objective function to be used with optuna."""

    @dataclasses.dataclass
    class TrialResults:
        id: int
        hyperparameters: dict
        validation_results: dict
        test_results: dict = None
        train_results: dict = None
        fit_time: float = None
        validation_time: float = None
        test_time: float = None
        algorithm: str = None

    @property
    def results(self):
        # By default, returns validation results
        return self.get_results("validation")

    def get_results(self, split: str = "validation"):
        _get_results_helper: callable
        if split in ("val", "validation"):
            _get_results_helper = lambda r: r.validation_results
        elif split in ("test", "testing"):
            _get_results_helper = lambda r: r.test_results
        elif split in ("train", "training"):
            _get_results_helper = lambda r: r.train_results
        else:
            raise ValueError(f"Value of type_='{split}' is invalid.")

        return pd.DataFrame(
            data=[
                {
                    "algorithm": r.algorithm,
                    **_get_results_helper(r),
                }
                for r in self._models_results
            ],
            index=[r.id for r in self._models_results],
        )

    @property
    def all_results(self):
        return self._models_results

    @property
    def best_trial(self):
        results = self.results.copy()
        target_metric_col = self.eval_metric

        if self.other_eval_metric:
            target_metric_col = "weighted_metric"
            results[target_metric_col] = results[
                self.eval_metric
            ] * self.alpha + results[self.other_eval_metric] * (1 - self.alpha)

        # NOTE: trial_idx != trial.id
        best_trial_idx = np.argmax(results[target_metric_col])
        return self.all_results[best_trial_idx]

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        threshold: Threshold,
        eval_metric: str,
        preprocessing_method: PreProcessing,
        inprocessing_method: InProcessing,
        postprocessing_method: PostProcessing,
        preprocessing_configs: Optional[DictConfig],
        inprocessing_configs: Optional[DictConfig],
        postprocessing_configs: Optional[DictConfig],
        s_train: pd.Series = None,
        s_val: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        s_test: pd.Series = None,
        other_eval_metric: Optional[str] = None,
        alpha: Optional[float] = 0.50,
        artifacts: Optional[list[str]] = (),
        artifacts_folder: Path = None,
    ):
        self.X_train, self.y_train, self.s_train = (X_train, y_train, s_train)
        self.X_val, self.y_val, self.s_val = (X_val, y_val, s_val)
        self.X_test, self.y_test, self.s_test = (X_test, y_test, s_test)

        self.preprocessing_method = preprocessing_method
        self.inprocessing_method = inprocessing_method
        self.postprocessing_method = postprocessing_method
        self.threshold = threshold

        self.preprocessing_configs = self.omegaconf_to_dict(preprocessing_configs)
        self.inprocessing_configs = self.omegaconf_to_dict(inprocessing_configs)
        self.postprocessing_configs = self.omegaconf_to_dict(postprocessing_configs)

        self.configs = self.join_configs(
            self.inprocessing_configs,
            self.preprocessing_configs,
            self.postprocessing_configs,
        )

        self.eval_metric = eval_metric
        self.other_eval_metric = other_eval_metric
        self.alpha = alpha
        assert alpha is None or (0 <= alpha <= 1)

        # Store all results in a list as models are trained
        self._models_results: list[ObjectiveFunction.TrialResults] = list()
        self.artifacts = artifacts
        self.artifacts_folder = artifacts_folder

    def __call__(self, trial: BaseTrial) -> float:
        if self.artifacts:
            trial_folder = self.artifacts_folder / str(trial.number)
            trial_folder.mkdir(parents=True, exist_ok=True)
        # Sample hyperparameters
        hyperparams = suggest_callable_hyperparams(trial, self.configs)
        hyperparams.pop("classpath", None)
        # Separate hyperparameters of each part
        (
            inprocessing_hyperparams,
            preprocessing_hyperparams,
            postprocessing_hyperparams,
        ) = self.separate_configs(hyperparams)

        # Construct model parts
        preprocessing = self.preprocessing_method(**preprocessing_hyperparams)
        inprocessing = self.inprocessing_method(**inprocessing_hyperparams)
        postprocessing = self.postprocessing_method(**postprocessing_hyperparams)

        # Train model
        start_time = time.process_time()
        preprocessing.fit(self.X_train, self.y_train, self.s_train)
        preprocessing_output = preprocessing.transform(
            self.X_train,
            self.y_train,
            self.s_train,
        )
        inprocessing.fit(*preprocessing_output)

        elapsed_process_time = time.process_time() - start_time
        logging.info(f"Trial {trial.number} took {elapsed_process_time}s to train.")

        # Evaluate model on validation data
        val_results = self.evaluate_model(
            model=(preprocessing, inprocessing, postprocessing, self.threshold),
            X=self.X_val,
            y=self.y_val,
            s=self.s_val,
            number=trial.number,
            type="validation",
        )
        if "methods" in self.artifacts:
            with open(trial_folder / "preprocessing.pickle", "wb") as file:
                pickle.dump(preprocessing, file)
            with open(trial_folder / "inprocessing.pickle", "wb") as file:
                pickle.dump(inprocessing, file)
            with open(trial_folder / "postprocessing.pickle", "wb") as file:
                pickle.dump(postprocessing, file)
            with open(trial_folder / "threshold.pickle", "wb") as file:
                pickle.dump(self.threshold, file)
        # Optionally, evaluate on test data as well (just to save results)
        test_results = None
        if self.X_test is not None and self.y_test is not None:
            test_results = self.evaluate_model(
                model=(preprocessing, inprocessing, postprocessing, self.threshold),
                X=self.X_test,
                y=self.y_test,
                s=self.s_test,
                number=trial.number,
                type="test",
            )

        # Store trial's results
        self._models_results.append(
            Result(
                id=trial.number,
                hyperparameters=hyperparams,
                validation_results=val_results,
                test_results=test_results,
                fit_time=elapsed_process_time,
            )
        )

        # Return scalarized evaluation metric
        if self.other_eval_metric:
            assert self.alpha is not None
            return val_results[self.eval_metric] * self.alpha + val_results[
                self.other_eval_metric
            ] * (1 - self.alpha)

        # Or simply a single evaluation metric value
        else:
            return val_results[self.eval_metric]

    def evaluate_model(
        self,
        model: tuple[PreProcessing, InProcessing, PostProcessing, Threshold],
        X: np.ndarray,
        y: np.ndarray,
        s: np.ndarray = None,
        number: int = 0,
        type: str = "validation",
    ) -> dict:
        if model[0].used_in_inference:
            X, y, s = model[0].transform(X.copy(), y.copy(), s.copy())
        y_pred = model[1].predict_proba(X, s)
        if "predictions" in self.artifacts:
            y_pred.to_frame().to_parquet(
                self.artifacts_folder / str(number) / f"{type}_scores.parquet"
            )
        if type == "validation":
            model[2].fit(X, y_pred, y, s)
        y_pred = model[2].transform(X, y_pred, s)
        if sorted(y_pred.unique().tolist()) != [0, 1]:
            if type == "validation":
                model[3].fit(X, y_pred, y, s)
            y_pred = model[3].transform(X, y_pred, s)
        if "predictions" in self.artifacts:
            y_pred.to_frame().to_parquet(
                self.artifacts_folder / str(number) / f"{type}_bin.parquet"
            )
        results = evaluate_performance(y, y_pred)

        if s is not None:
            results.update(evaluate_fairness(y, y_pred, s, True))

        return results

    @staticmethod
    def join_configs(*configs: dict) -> dict:
        result = {}
        for index, config in enumerate(configs):
            if config is not None:
                result.update(
                    {f"{index}_{key}": value for key, value in config.items()}
                )

        return {"algorithm": {"kwargs": result, "classpath": ""}}

    @staticmethod
    def separate_configs(configs: dict) -> tuple[dict]:
        results = []
        max_iter = max([int(key.split("_")[0]) for key in configs.keys()])
        max_iter = 2 if max_iter < 2 else max_iter

        for index in range(max_iter + 1):
            temp_dict = {}
            for key, value in configs.items():
                if key.startswith(str(index) + "_"):
                    temp_dict.update({"_".join(key.split("_")[1:]): value})
            results.append(temp_dict)

        return tuple(results)

    @staticmethod
    def omegaconf_to_dict(config: Optional[DictConfig]) -> Union[dict, None]:
        if config is not None:
            return OmegaConf.to_container(config)
        else:
            return None
