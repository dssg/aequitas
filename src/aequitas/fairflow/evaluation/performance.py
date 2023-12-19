import pandas as pd

from .fairness import METRICS, evaluate_fairness


def evaluate_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Evaluates performance of a machine learning model.

    To utilize the Aequitas framework, we set the sensitive attribute to a constant.

    Parameters
    ----------
    y_true : pd.Series
        The true class labels.
    y_pred : pd.Series
        The binarized predictions.

    Returns
    -------
    dict
        Dictionary with performance metrics.
    """
    result = {}

    sensitive_attribute = pd.Series([1] * len(y_true), index=y_true.index).astype(str)
    fairness_results = evaluate_fairness(
        y_true,
        y_pred,
        sensitive_attribute,
        return_groupwise_metrics=True,
    )

    for metric in METRICS:
        result[metric] = fairness_results[f"{metric}_1"]

    return result
