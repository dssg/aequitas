from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class PostProcessing(ABC):
    """
    Abstract class for a wrapper of a postprocessing technique.

    Methods
    -------
    fit(X, y_hat, y, s)
        Fit the post-processing technique to the validation data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        y_hat : pandas.Series
            The pre-transformed predictions.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The protected attribute.

    transform(X, y_hat, s)
        Transform the predictions according to the technique.

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
        pandas.Series
            The transformed predicted values.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> None:
        """
        Fit the post-processing technique to the validation data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        y_hat : pandas.Series
            The pre-transformed predictions.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The protected attribute.
        """
        pass

    @abstractmethod
    def transform(
        self,
        X: pd.DataFrame,
        y_hat: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Transform the predictions according to the technique.

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
        pandas.Series
            The transformed predicted values.
        """
        pass
