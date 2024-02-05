from typing import Optional

import numpy as np

from ...utils.evaluation import bootstrap_hyperparameters
from ...utils.metrics import METRIC_NAMES, FAIRNESS_METRIC, PERFORMANCE_METRIC
from ...evaluation import Result


DEFAULT_KWARGS = {
    "bootstrap_size": 0.2,
    "alpha_points": np.arange(0, 1.01, 0.01),
    "evaluate_on": "test",
    "n_trials": 100,
}


class Plot:
    def __init__(
        self,
        results: dict[str, dict[str, Result]],
        dataset: str,
        fairness_metric: FAIRNESS_METRIC,
        performance_metric: PERFORMANCE_METRIC,
        method: Optional[str] = None,
        confidence_intervals: float = 0.95,
        **kwargs,
    ):
        self.method = method
        if dataset in results:
            self.dataset = dataset
        else:
            raise ValueError(
                f"Dataset {dataset} not found in results. "
                f"Use one of {list(results.keys())}"
            )

        if method:
            # Keeping the data format consistent
            self.raw_results = {method: results[dataset][method]}
        else:
            self.raw_results = results[dataset]

        self.dataset = dataset
        self.method = method
        self.fairness_metric = fairness_metric
        self.performance_metric = performance_metric
        self.confidence_intervals = confidence_intervals
        self.kwargs = kwargs
        # Update self.kwargs with default values if not populated
        for key, value in DEFAULT_KWARGS.items():
            if key not in self.kwargs:
                self.kwargs[key] = value
        self.kwargs["fairness_metric"] = METRIC_NAMES[fairness_metric]
        self.kwargs["performance_metric"] = METRIC_NAMES[performance_metric]
        self.bootstrap_results = {}
        if isinstance(self.kwargs["alpha_points"], np.ndarray):
            self.x = self.kwargs["alpha_points"]
            self.plot_type = "alpha"
        else:
            self.x = self.kwargs["bootstrap_size"]
            self.plot_type = "bootstrap"

    def _generate_bootstrap_samples(self):
        if self.method:
            self.bootstrap_results[self.method] = bootstrap_hyperparameters(
                self.raw_results, **self.kwargs
            )
        else:
            for method, results in self.raw_results.items():
                self.bootstrap_results[method] = bootstrap_hyperparameters(
                    results, **self.kwargs
                )

    def visualize(self):
        from .visualize import visualize

        if not self.bootstrap_results:
            self._generate_bootstrap_samples()

        visualize(self)
