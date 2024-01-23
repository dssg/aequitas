from typing import Optional

import numpy as np
import pandas as pd

from aequitas.flow.methods.preprocessing.preprocessing import PreProcessing


class DataRepairer(PreProcessing):
    def __init__(
        self,
        repair_level: float = 1.0,
        columns: Optional[list[str]] = None,
        definition: int = 101,
    ):
        """
        Transforms the data distribution so that a given feature distribution is
        more or less independent of the sensitive attribute s.

        This is achieved by matching the conditional distribution P(X|s) to the
        global variable distribution P(X), matching the values of quantiles.

        Parameters
        ----------
        repair_level: float
            How much will the data be transformed to the global distribution.
            Defaults to 1.0.
        columns: list[str], optional
            Which columns to transform. If left empty, transforms all columns of
            X.
        definition: int
            How many quantiles to calculate. Defaults to 101.

        Attributes
        ----------
        _quantile_points: np.ndarray
            Values of quantiles. The length is determined by the definition.
        _global_quantiles: dict[str, numpy.ndarray]
            Quantiles of the features to transform.
        _group_quantiles: dict[str, dict[str], numpy.ndarray]
            Quantiles of the features to transform depending on the group.

        Methods
        -------
        fit(X, y_hat, y, s=None)
            Calculates the quantiles in the dataset for each feature and group.

        transform(X, y_hat, s=None)
            Transform the features to match the global distribution.

        These methods are the ones to be implemented, which are defined by the
        parent abstract class.
        """
        self.repair_level = repair_level
        self.columns = columns
        self.definition = definition

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """
        Calculates the quantiles in the dataset for each feature and group.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series
            The labels. Note that this is not used by the method but we must
            follow the parent class method signature in the example.
        s : pd.Series, optional
            The sensitive attribute.
        """
        super().fit(X, y, s)

        if self.columns is None:
            self.columns = X.columns.tolist()
        if s is None:
            raise ValueError("s must be passed.")
        self._quantile_points = np.linspace(0, 1, self.definition)
        # Create a dictionary with quantile values per group.
        self._group_quantiles = {}
        # Also create a dictionary with the global quantiles.
        self._global_quantiles = {}
        for column in self.columns:
            self._global_quantiles[column] = (
                X[column].quantile(self._quantile_points).values
            )

            # Get the quantiles for each group in
            self._group_quantiles[column] = self._get_group_quantiles(X, s, column)

    def _get_group_quantiles(
        self,
        X: pd.DataFrame,
        s: pd.Series,
        column: str,
    ) -> dict[str, np.ndarray]:
        """
        Transforms the quantiles to a more digested value.

        The original method of pandas creates a dictionary with a two-level
        index, which is harder to query. This transforms the two level index in
        dictionaries within a single dictionary.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix.
        s : pd.Series
            The sensitive attribute.
        column : str
            The feature to calculate the quantiles.
        """
        quantile_dict = X.groupby(s)[column].quantile(self._quantile_points)
        transformed_quantile_dict = {}
        for group in quantile_dict.index.get_level_values(0).unique():
            transformed_quantile_dict[group] = quantile_dict[group].values
        return transformed_quantile_dict

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        s: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Transform the features conditioned of protected attribute to match
        the global distribution.

        Parameters
        ----------
        X : pd.DataFrame
            The features.
        y : pandas.Series
            The labels.
        s : pandas.Series, optional
            The protected attribute.

        Returns
        -------
        pd.DataFrame, pd.Series, pd.Series
            Transformed features, labels, and sensitive attribute.
        """
        super().transform(X, y, s)
        
        if s is None:
            raise ValueError("s must be passed.")

        X_repaired = X.copy()

        for column in self.columns:
            # Calculate the quantile of every point for every group (vectorized)
            interpolation_quantiles = {}
            # Also calculate the global distribution value at that quantile
            global_values = {}
            # After, calculate the corrected value for every point
            corrected_values = {}
            for group in s.unique():
                interpolation_quantiles[group] = (
                    np.interp(
                        X_repaired[column],
                        self._group_quantiles[column][group],
                        self._quantile_points,
                    )
                    + np.interp(
                        -X_repaired[column],
                        -self._group_quantiles[column][group][::-1],
                        self._quantile_points[::-1],
                    )
                ) / 2
                global_values[group] = np.interp(
                    interpolation_quantiles[group],
                    self._quantile_points,
                    self._global_quantiles[column],
                )
                corrected_values[group] = global_values[
                    group
                ] * self.repair_level + X_repaired[column].values * (
                    1 - self.repair_level
                )

            repaired_column = [
                corrected_values[group][index] for index, group in enumerate(s)
            ]
            X_repaired[column] = repaired_column
        return X_repaired, y, s
