from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class PreProcessing(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """
        Fit the preprocessing method to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            The target values.
        s : pd.Series, optional
            The protected attribute.
        """
        pass

    @abstractmethod
    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Transform the data according to the preprocessing method.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            The target values.
        s : Optional[pd.Series], optional
            The protected attribute.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        pass
