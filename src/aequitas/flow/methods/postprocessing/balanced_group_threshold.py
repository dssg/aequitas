from typing import Optional, Union

import numpy as np
import pandas as pd

from ...utils import create_logger
from .postprocessing import PostProcessing
from .threshold import Threshold


class BalancedGroupThreshold(PostProcessing):
    def __init__(
        self,
        threshold_type: str,
        threshold_value: Union[float, int],
        fairness_metric: str,
    ):
        """Initialize a new instance of the BalancedGroupThreshold class.

        Parameters
        ----------
        threshold_type : str
            The type of threshold to apply. It can be one of the following:
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

    def fit(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ):
        """Fit a threshold for each group in the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y_hat : pd.Series
            The pre-transformed predictions.
        y : pd.Series
            The target values.
        s : pd.Series, optional
            The protected attribute.
        """
        unique_groups = s.unique()

        def process_group(group_df):
            group_df.sort_values(by="y_hat", ascending=False, inplace=True)
            if self.fairness_metric == "fpr":
                relevant_labels = [0]
            elif self.fairness_metric == "tpr":
                relevant_labels = [1]
            elif self.fairness_metric == "pprev":
                relevant_labels = [0, 1]
            else:
                raise ValueError(
                    f"Fairness metric {self.fairness_metric} is not supported."
                )

            n_relevant_labels = (group_df["y"].isin(relevant_labels)).sum()
            relevant_ids = group_df[group_df["y"].isin(relevant_labels)].index
            vals = np.concatenate(
                [
                    np.linspace(
                        0,
                        1,
                        n_relevant_labels,
                        endpoint=False,
                    ),
                    np.array([1]),
                ]
            )

            # Create a mask for the DataFrame
            mask = group_df.index.isin(relevant_ids)

            # Perform operations on the DataFrame using the mask
            group_df.loc[mask, "value"] = vals[1:]
            # Forward fill the 'value' column
            group_df["value"].fillna(method="ffill", inplace=True)
            group_df["value"].fillna(0, inplace=True)
            return group_df

        # Create a single DataFrame
        df = pd.DataFrame(
            {
                "y_hat": y_hat,
                "y": y,
                "s": s,
            }
        )
        # Add a 'group' column to the DataFrame
        df["group"] = s

        # Use groupby and apply to process each group
        all_groups_df = (
            df.groupby("group", group_keys=False)
            .apply(process_group)
            .sort_values("value")
        )

        if self.threshold_type == "top_pct":
            pos_predictions = int(all_groups_df.shape[0] * self.threshold_value)
            neg_predictions = all_groups_df.shape[0] - pos_predictions
            predictions = [1] * pos_predictions + [0] * neg_predictions
            all_groups_df["predictions"] = predictions

        elif self.threshold_type == "top_k":
            pos_predictions = self.threshold_value
            neg_predictions = all_groups_df.shape[0] - pos_predictions
            predictions = [1] * pos_predictions + [0] * neg_predictions
            all_groups_df["predictions"] = predictions

        elif self.threshold_type == "fpr":
            ln_values = np.array(all_groups_df[all_groups_df["y"] == 0]["value"].values)
            threshold = np.percentile(ln_values, (1 - self.threshold_value) * 100)
            all_groups_df["predictions"] = (all_groups_df["value"] <= threshold).astype(
                int
            )

        elif self.threshold_type == "tpr":
            ln_values = np.array(all_groups_df[all_groups_df["y"] == 1]["value"].values)
            threshold = np.percentile(ln_values, self.threshold_value * 100)
            all_groups_df["predictions"] = (all_groups_df["value"] <= threshold).astype(
                int
            )

        for group in unique_groups:
            group_df = all_groups_df[all_groups_df["s"] == group]
            idx = group_df[group_df["predictions"].diff() < 0].index[0]
            pos = np.where(group_df.index.to_numpy() == idx)[0][0]
            prev_idx = group_df.index[pos - 1]
            threshold = group_df.loc[[prev_idx, idx]]["y_hat"].mean()
            self.thresholds[group] = Threshold("fixed", threshold)

    def transform(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        s: Optional[pd.Series] = None,
    ):
        """Transform the prediction scores based on the threshold for each group in the
        dataset.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        y_hat : pandas.Series
            The pre-transformed predictions.
        s : pandas.Series, optional
            The protected attribute.

        Returns
        -------
        pd.Series
            Transformed predicted scores.
        """
        if s is None:
            raise ValueError("`s` must be provided to transform with a GroupThreshold.")
        predictions = []
        for group in s.unique():
            predictions.append(
                self.thresholds[group].transform(
                    X[s == group],
                    y_hat[s == group],
                    s[s == group],
                )
            )
        return pd.concat(predictions)
