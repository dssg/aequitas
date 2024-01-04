from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class InProcessing(ABC):
    """
    Abstract class for a wrapper of an in-processing FairML model.

    Methods
    -------
    fit(X, y, s)
        Train the machine learning model on the provided data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The protected attribute.

    predict_proba(X, s)
        Use the machine learning model to make predictions on new data.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        s : pandas.Series, optional
            The protected attribute.

        Returns
        -------
        pandas.Series
            The predicted values.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the machine learning model on the provided data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The protected attribute.
        """
        pass

    @abstractmethod
    def predict_proba(
        self, X: pd.DataFrame, s: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Use the machine learning model to make predictions on new data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        s : pandas.Series, optional
            The protected attribute.
        Returns
        -------
        pandas.Series
            The predicted values.
        """
        pass
