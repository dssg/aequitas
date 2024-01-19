import dataclasses

from .fairness import evaluate_fairness
from .performance import evaluate_performance

__all__ = ["evaluate_fairness", "evaluate_performance"]


@dataclasses.dataclass
class Result:
    """Result of a benchmark run.

    Parameters
    ----------
    fairness_metrics : dict
        Dictionary with fairness metrics.
    performance_metrics : dict
        Dictionary with performance metrics.
    """

    fairness_metrics: dict[str, float]
    performance_metrics: dict[str, float]
