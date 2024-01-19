from typing import Optional

import pandas as pd

from ...group import Group

METRICS = list(Group.all_group_metrics) + [
    "fp",
    "fn",
    "tn",
    "tp",
]


def evaluate_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attribute: pd.Series,
    return_groupwise_metrics: Optional[bool] = False,
) -> dict:
    """Evaluates fairness as the ratios or differences between group-wise performance
    metrics.

    Parameters
    ----------
    y_true : pd.Series
        The true class labels.
    y_pred : pd.Series
        The binarized predictions.
    sensitive_attribute : pd.Series
        The sensitive attribute (protected group membership).

    Returns
    -------
    dict
        Dictionary with fairness metrics.
    """
    result = {}

    group = Group()

    df = pd.concat([y_true, y_pred, sensitive_attribute.astype(str)], axis=1).copy()
    df.columns = ["label_value", "score", "sensitive_attribute"]

    metrics_df, _ = group.get_crosstabs(df)

    for metric in METRICS:
        max_metric = metrics_df[metric].max()
        min_metric = metrics_df[metric].min()
        result[f"{metric}_diff"] = max_metric - min_metric
        result[f"{metric}_ratio"] = min_metric / max_metric

    if return_groupwise_metrics:
        unique_groups = sensitive_attribute.unique()
        for group in unique_groups:
            group_df = metrics_df[metrics_df["attribute_value"] == str(group)]
            for metric in METRICS:
                result[f"{metric}_{group}"] = group_df[metric].values[0]

    return result
