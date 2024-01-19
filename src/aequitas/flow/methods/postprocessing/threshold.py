from typing import Optional, Union

import numpy as np
import pandas as pd

from ...utils import create_logger
from .postprocessing import PostProcessing

THRESHOLD_TYPES = ["fixed", "fpr", "tpr", "top_pct", "top_k"]


class Threshold(PostProcessing):
    """A postprocessing class that applies thresholding to the model's predictions.

    Parameters
    ----------
    threshold_type : str
        Type of threshold to apply to the model's predictions. Must be one of:
        'fixed', 'fpr', 'tpr', 'top_pct', 'top_k'.
    threshold_value : Union[float, int]
        Threshold value to apply. The meaning of the threshold value depends on
        the threshold_type. Must be between 0 and 1 for 'fixed', 'fpr', and
        'tpr' threshold types. Must be an integer for 'top_k' threshold type.

    Methods
    -------
    fit(X, y_hat, y, s=None)
        Compute the threshold based on the given threshold_type and threshold_value.
    """

    def __init__(self, threshold_type: str, threshold_value: Union[float, int]):
        """Initialize a new Threshold instance.

        Parameters
        ----------
        threshold_type : str
            Type of threshold to apply to the model's predictions. Must be one of:
            'fixed', 'fpr', 'tpr', 'top_pct', 'top_k'.
        threshold_value : Union[float, int]
            Threshold value to apply. The meaning of the threshold value depends on
            the threshold_type. Must be between 0 and 1 for 'fixed', 'fpr', and
            'tpr' threshold types. Must be an integer for 'top_k' threshold type.

        Raises
        ------
        ValueError
            If an invalid threshold_type or threshold_value is provided.
        """
        self.logger = create_logger("methods.postprocessing.Threshold")
        self.logger.info("Instantiating postprocessing Threshold.")
        # Validate type
        if threshold_type not in THRESHOLD_TYPES:
            raise ValueError(f"Invalid threshold type. Select one of {THRESHOLD_TYPES}")
        self.threshold_type = threshold_type
        # Validate value
        if self.threshold_type in THRESHOLD_TYPES[:-1]:  # Assuming THRESHOLD_TYPES
            if 0 <= threshold_value <= 1:
                pass
            else:
                raise ValueError("Invalid threshold value, must be between 0 and 1.")
        else:
            if not isinstance(threshold_value, int):
                raise ValueError("Invalid threshold value, must be integer for top_k.")
        self.threshold_value = threshold_value
        self.logger.debug(
            f"{self.threshold_type} Threshold with value {self.threshold_value}"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> None:
        """Compute the threshold based on the given threshold_type and threshold_value.

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
        self.logger.info("Computing threshold.")

        if self.threshold_type in [
            "fixed",
            "top_pct",
            "top_k",
        ]:  # Fitted in transformation step
            pass

        elif self.threshold_type == "fpr":
            ln_scores = np.array(y_hat[y == 0].values)
            self.threshold = np.percentile(ln_scores, (1 - self.threshold_value) * 100)
            self.logger.debug(f"Threshold of value {self.threshold}")

        elif self.threshold_type == "tpr":
            lp_scores = np.array(y_hat[y == 1].values)
            self.threshold = np.percentile(lp_scores, self.threshold_value * 100)
            self.logger.debug(f"Threshold of value {self.threshold}")

        self.logger.info("Finished computing threshold.")

    def transform(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Transform predicted scores using threshold.

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
        self.logger.info("Transforming predictions.")

        if self.threshold_type == "fixed":
            y_pred = (y_hat >= self.threshold_value).astype(int)
        elif self.threshold_type in ["fpr", "tpr"]:
            y_pred = (y_hat >= self.threshold).astype(int)
        elif self.threshold_type == "top_pct":
            n_top = int(len(y_hat) * self.threshold_value)
            y_pred = (y_hat >= y_hat.nlargest(n_top).min()).astype(int)
        elif self.threshold_type == "top_k":
            # Check if threshold value is smaller than number of samples
            if self.threshold_value > len(y_hat):
                raise ValueError(
                    f"Threshold value {self.threshold_value} is larger "
                    f"than the number of samples {len(y_hat)}."
                )
            y_pred = (y_hat >= y_hat.nlargest(self.threshold_value).min()).astype(int)
        else:
            raise ValueError(f"Invalid threshold type: {self.threshold_type}")
        self.logger.info("Finished transforming predictions.")
        return y_pred
