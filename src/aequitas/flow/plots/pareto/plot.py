from typing import Any, Literal, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from ...evaluation import Result
from ....bias import Bias
from ....group import Group
from ....plot import summary, disparity


_names = {
    "exponentiated_gradient_baf": "EG",
    "exponentiated_gradient_folktables": "EG",
    "grid_search_baf": "GS",
    "grid_search_folktables": "GS",
    "fairgbm_baf": "FairGBM",
    "fairgbm_folktables": "FairGBM",
    "group_threshold_baf": "Thresholding",
    "group_threshold_folktables": "Thresholding",
    "lightgbm_baseline": "LightGBM",
    "prevalence_oversampling": "Oversampling",
    "prevalence_under_sampling": "Undersampling",
}


def _prettify_names(method_name: str) -> str:
    """Prettifies the method name for plotting.

    Parameters
    ----------
    method_name : str
        The name of the method.

    Returns
    -------
    str
        The prettified name of the method.
    """
    return _names.get(method_name, method_name)


class Plot:
    """Class for the visualization of Pareto models. Handles the intermediary steps
    between having the results and creating the plot.

    Parameters
    ----------
    results : dict[str, dict[str, Result]]
        A dictionary of results, obtained from
        aequitas.fairflow.utils.artifacts.read_results.
    dataset : str
        Name of the dataset to be used in the Pareto plot.
    method : union[str, list], optional
        Name of the method to plot. If none, all methods will be plotted.
    fairness_metric : {"Predictive Equality", "Equal Opportunity", "Demographic Parity"}
        The default fairness metric to use in the Pareto plot.
    performance_metric : {"TPR", "FPR", "FNR", "Accuracy", "Precision"}
        The default performance metric to use in the Pareto plot.
    alpha : float, optional
        The alpha value to use in the Pareto plot.
    direction : {"minimize", "maximize"}, optional
        The direction to use in the Pareto plot.
    """

    def __init__(
        self,
        results: dict[str, dict[str, Result]],
        dataset: str,
        fairness_metric: Literal[
            "Predictive Equality", "Equal Opportunity", "Demographic Parity"
        ],
        performance_metric: Literal["TPR", "FPR", "FNR", "Accuracy", "Precision"],
        method: Optional[Union[str, list]] = None,
        alpha: float = 0.5,
        direction: Literal["minimize", "maximize"] = "maximize",
        split: Literal["validation", "test"] = "test",
    ):
        self.method = method
        if dataset in results:
            self.dataset = dataset
        else:
            raise ValueError(
                f"Dataset {dataset} not found in results."
                f"Use one of {list(results.keys())}"
            )

        if method is None:
            raw_results = results[dataset]
        elif isinstance(method, str):
            raw_results = {method: results[dataset][method]}
        elif isinstance(method, list):
            raw_results = {m: results[dataset][m] for m in method}
        else:
            raise ValueError(
                f"Method {method} not recognized. Please use a string, a list of "
                f"strings, or leave the field empty."
            )

        self.split = split
        # Cast the results to the desired format
        self._results = [
            self._dataclass_to_dict(res, _prettify_names(method), method, self.split)
            for method, results in raw_results.items()
            for res in results
        ]
        self.fairness_metric = fairness_metric
        self.performance_metric = performance_metric
        self.alpha = alpha
        self.direction = direction
        self.available_fairness_metrics = {
            "Predictive Equality",
            "Equal Opportunity",
            "Demographic Parity",
        }  # Hardcoded for now
        self.available_performance_metrics = [
            "TPR",
            "FPR",
            "FNR",
            "Accuracy",
            "Precision",
        ]
        self._best_model_idx: int = 0

    @property
    def results(self):
        df = pd.DataFrame(self._results).set_index("model_id")
        df["alpha"] = (
            df[self.fairness_metric] * (1 - self.alpha)
            + df[self.performance_metric] * self.alpha
        )
        df = df.reset_index()
        df["model_id"] = list(range(df.shape[0]))
        # Update best model id
        self._best_model_idx = df[df["alpha"] == df["alpha"].max()].index[0]
        df.drop(columns="model", inplace=True, errors="ignore")
        return df

    @property
    def best_model_details(self):
        dct = self.results.iloc[self._best_model_idx].to_dict().copy()
        dct.pop("model", None)
        return dct

    def _compute_pareto_models(self):
        metrics = [self.performance_metric]
        if self.fairness_metric:
            metrics.append(self.fairness_metric)

        points = self.results[metrics].to_numpy()
        is_pareto = self._is_pareto_efficient(points, direction=self.direction)
        for idx in range(len(self._results)):
            self._results[idx]["is_pareto"] = is_pareto[idx]

    @staticmethod
    def _is_pareto_efficient(points: np.ndarray, direction: str = "minimize"):
        """Finds the pareto-efficient points.

        Parameters
        ----------
        points : np.ndarray
            An (n_points, n_costs) array with the costs if direction=="minimize", or
            objectives to maximize if direction=="maximize".
        direction : "minimize" or "maximize", optional (default: "minimize")
            Whether to minimize or maximize the given points.

        Returns
        -------
        List[bool]
            A boolean np.ndarray with the same length as the input array, and whose
            value indicates if the model at a given index is Pareto efficient.
        """
        assert direction in {"minimize", "maximize"}
        if direction == "maximize":
            points = 1 - points
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    points[is_efficient] < c, axis=1
                )  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    @staticmethod
    def _dataclass_to_dict(dataclass, method, original_method_name, split):
        hyperparameters = dataclass.hyperparameters
        if split == "validation":
            results = dataclass.validation_results
        elif split == "test":
            results = dataclass.test_results
        else:
            raise ValueError(f"Split {split} not recognized.")
        hyperparameters.update({"classpath": method})
        # if hyperparameter has value of None, change it to string "None"
        for key, value in hyperparameters.items():
            if value is None:
                hyperparameters[key] = "None"
        return {
            "model_id": dataclass.id,
            "internal_id": dataclass.id,
            "internal_method_name": original_method_name,
            "TPR": results["tpr"],
            "FPR": results["fpr"],
            "FNR": results["fnr"],
            "Accuracy": results["accuracy"],
            "Precision": results["precision"],
            "Equal Opportunity": results["tpr_ratio"],
            "Predictive Equality": results["fpr_ratio"],
            "Demographic Parity": results["pprev_ratio"],
            "algorithm": hyperparameters["classpath"],
            "hyperparams": hyperparameters,
        }

    def visualize(self, **kwargs):
        """Render interactive application to explore results of hyperparameter
        optimization search.
        """

        from .visualize import visualize

        return visualize(self, **kwargs)

    def bias_audit(
        self,
        model_id: int,
        dataset: Any,
        sensitive_attribute: Union[str, list[str]],
        metrics: list[str] = ["tpr", "fpr"],
        fairness_threshold: float = 1.2,
        results_path: Union[Path, str] = "examples/experiment_results",
        reference_groups: Optional[dict[str, str]] = None,
    ):
        """Render interactive application to audit bias of a given model.

        Parameters
        ----------
        model_id : int
            The id of the model to audit.
        """
        if isinstance(results_path, str):
            results_path = Path(results_path)

        if isinstance(sensitive_attribute, str):
            sensitive_attribute = [sensitive_attribute]

        method_name = self.results.iloc[model_id]["internal_method_name"]
        method_id = self.results.iloc[model_id]["internal_id"]

        predictions_path = (
            results_path
            / self.dataset
            / method_name
            / str(method_id)
            / "test_bin.parquet"
        )

        # Check if predictions path exists
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions for model {method_name} with id {method_id} not found. "
                "Please make sure that the predictions are stored in the correct path."
            )

        # Read the predictions
        predictions = pd.read_parquet(predictions_path)

        # Add the predictions to the DataFrame
        label = dataset.y.name
        dataset = dataset.copy()
        dataset["predictions"] = predictions

        dataset[sensitive_attribute] = dataset[sensitive_attribute].astype(str)

        g = Group()
        b = Bias()

        cm_metrics, _ = g.get_crosstabs(
            df=dataset,
            score_col="predictions",
            label_col=label,
            attr_cols=sensitive_attribute,
        )

        if not reference_groups:
            reference_groups = {
                attr: str(dataset[attr].mode().values[0])
                for attr in sensitive_attribute
            }

        disparity_metrics = b.get_disparity_predefined_groups(
            cm_metrics, dataset, ref_groups_dict=reference_groups
        )

        return summary(
            disparity_metrics, metrics, fairness_threshold=fairness_threshold
        )

    def disparities(
        self,
        model_id: int,
        dataset: Any,
        sensitive_attribute: Union[str, list[str]],
        metrics: list[str] = ["tpr", "fpr"],
        fairness_threshold: float = 1.2,
        results_path: Union[Path, str] = "examples/experiment_results",
        reference_groups: Optional[dict[str, str]] = None,
    ):
        """Render interactive application to audit bias of a given model.

        Parameters
        ----------
        model_id : int
            The id of the model to audit.
        """
        if isinstance(results_path, str):
            results_path = Path(results_path)

        if isinstance(sensitive_attribute, str):
            sensitive_attribute = [sensitive_attribute]

        method_name = self.results.iloc[model_id]["internal_method_name"]
        method_id = self.results.iloc[model_id]["internal_id"]

        predictions_path = (
            results_path
            / self.dataset
            / method_name
            / str(method_id)
            / "test_bin.parquet"
        )

        # Check if predictions path exists
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions for model {method_name} with id {method_id} not found. "
                "Please make sure that the predictions are stored in the correct path."
            )

        # Read the predictions
        predictions = pd.read_parquet(predictions_path)

        # Add the predictions to the DataFrame
        label = dataset.y.name
        dataset = dataset.copy()
        dataset["predictions"] = predictions

        dataset[sensitive_attribute] = dataset[sensitive_attribute].astype(str)

        g = Group()
        b = Bias()

        cm_metrics, _ = g.get_crosstabs(
            df=dataset,
            score_col="predictions",
            label_col=label,
            attr_cols=sensitive_attribute,
        )

        if not reference_groups:
            reference_groups = {
                attr: str(dataset[attr].mode().values[0])
                for attr in sensitive_attribute
            }

        disparity_metrics = b.get_disparity_predefined_groups(
            cm_metrics, dataset, ref_groups_dict=reference_groups
        )

        return disparity(
            disparity_metrics,
            metrics,
            sensitive_attribute[0],
            fairness_threshold=fairness_threshold,
        )
