from typing import Optional, Union

import pandas as pd

from ...utils import create_logger
from .postprocessing import PostProcessing


class GroupThreshold(PostProcessing):
    def __init__(
        self,
        threshold_type: str,
        threshold_value: Union[float, int],
        fairness_metric: str,
    ):
        """Initialize a new instance of the GroupThreshold class.

        Parameters
        ----------
        threshold_type : str
            The type of threshold to apply. It can be one of the following:
                - fixed: applies a fixed threshold for all samples.
                - fpr: applies a threshold to obtain a specific false positive rate.
                - tpr: applies a threshold to obtain a specific true positive rate.
                - top_pct: applies a threshold to obtain the top percentage of
                           predicted scores.
                - top_k: applies a threshold to obtain the top k predicted scores.
        threshold_value : Union[float, int]
            The value to use for the threshold, depending on the threshold_type
            parameter.
        fairness_metric : str
            The metric to use for measurement of fairness. It can be one of the 
            following:
                - tpr: true positive rate
                - fpr: false positive rate
                - pprev: predicted prevalence
        """
        self.logger = create_logger("methods.postprocessing.BalancedGroupThreshold")
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.fairness_metric = fairness_metric

        self.thresholds = {}

    def fit(self, X, y_hat, y, s=None):
        """Fit a threshold for each group in the dataset.

        Parameters
        ----------
        X : array-like
            The input samples.
        y_hat : array-like
            The predicted scores.
        y : array-like
            The target values.
        s : array-like, optional
            The group identifiers for the samples.
        """
        unique_groups = s.unique().values

        weight_fairness = []
        # We will create a vector of the fairness metric w.r.t. the order of scoring.
        # This will be used to compute the threshold for each group.
        for group in unique_groups:
            group_mask = s == group
            group_y_hat = y_hat[group_mask]
            group_y = y[group_mask]
            group_s = s[group_mask]
            group_df = pd.DataFrame(
                {
                    "y_hat": group_y_hat,
                    "y": group_y,
                    "s": group_s,
                }
            )
            group_df.sort_values(by="y_hat", ascending=False, inplace=True)
            
            if fairness_metric == "tpr":
                weight_fairness.append(
                    
                )


    def transform(self, X, y_hat, s=None):
        """Transform the prediction scores based on the threshold for each group in the
        dataset.

        Parameters
        ----------
        X : array-like
            The input samples.
        y_hat : array-like
            The predicted scores.
        s : array-like, optional
            The group identifiers for the samples.
        """
        # TODO: Implement the method to adjust the prediction scores.