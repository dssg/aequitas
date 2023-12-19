from typing import Optional, Union

import pandas as pd

from ...utils import create_logger
from .threshold import Threshold


class GroupThreshold(Threshold):
    """A post-processing class to adjust the prediction scores based on a threshold for
    multiple groups in the dataset.

    Parameters
    ----------
    threshold_type : str
        The type of threshold to apply. It can be one of the following:
            - fixed: applies a fixed threshold for all samples.
            - fpr: applies a threshold to obtain a specific false positive rate.
            - tpr: applies a threshold to obtain a specific true positive rate.
            - top_pct: applies a threshold to obtain the top percentage of predicted
                       scores.
            - top_k: applies a threshold to obtain the top k predicted scores.
    threshold_value : Union[float, int]
        The value to use for the threshold, depending on the threshold_type parameter.

    Attributes
    ----------
    thresholds : dict[str, Threshold]
        A dictionary of Threshold objects, one for each group in the dataset.

    Methods
    -------
    fit(X, y_hat, y, s=None)
        Fit the threshold for each group in the dataset.

    transform(X, y_hat, s=None)
        Transform the prediction scores based on the threshold for each group in the
        dataset.

    """

    def __init__(
        self,
        threshold_type: str,
        threshold_value: Union[float, int],
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
        """
        super().__init__(threshold_type, threshold_value)
        self.logger = create_logger("methods.postprocessing.GroupThreshold")
        self.logger.info("Instantiating postprocessing GroupThreshold.")
        self.thresholds: dict[str, Threshold] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> None:
        """Fit a threshold for each group in the dataset.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        y_hat : pd.Series
            The predicted scores.
        y : pd.Series
            The true labels.
        s : pd.Series, optional
            The sensitive attribute used to group the samples.
        """
        if s is None:
            raise ValueError("`s` must be provided to fit a GroupThreshold.")
        for group in s.unique():
            self.thresholds[group] = Threshold(
                self.threshold_type, self.threshold_value
            )
            self.thresholds[group].fit(
                X[s == group],
                y_hat[s == group],
                y[s == group],
                s[s == group],
            )

    def transform(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> pd.Series:
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
