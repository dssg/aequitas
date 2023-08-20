import numpy as np
import pandas as pd

from typing import Any, Literal


class ParetoWrapper:
    """Wrapper class for the visualization of Pareto models.

    Parameters
    ----------
    """

    def __init__(
        self,
        results: list[dict[str, list[Any]]],
        fairness_metric: Literal[
            "Predictive Equality", "Equal Opportunity", "Demographic Parity"
        ],
        performance_metric,
        method,
        alpha: float = 0.5,
        direction: Literal["minimize", "maximize"] = "maximize",
    ):
        self.method = method
        # Cast the results to the desired format
        self._results = [self._dataclass_to_dict(res, method) for res in results]
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
    def _dataclass_to_dict(dataclass, method):
        hyperparameters = dataclass.hyperparameters
        hyperparameters.update({"classpath": method})
        return {
            "model_id": dataclass.id,
            "TPR": dataclass.test_results["tpr"],
            "FPR": dataclass.test_results["fpr"],
            "FNR": dataclass.test_results["fnr"],
            "Accuracy": dataclass.test_results["accuracy"],
            "Precision": dataclass.test_results["precision"],
            "Equal Opportunity": dataclass.test_results["tpr_ratio"],
            "Predictive Equality": dataclass.test_results["fpr_ratio"],
            "Demographic Parity": dataclass.test_results["ppr_ratio"],
            "algorithm": hyperparameters["classpath"],
            "hyperparams": hyperparameters,
        }
