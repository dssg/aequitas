from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from ...utils import create_logger
from .preprocessing import PreProcessing


class CorrelatedFeatureRemoval(PreProcessing):
    def __init__(self, correlation_threshold: Optional[float] = 0.5):
        """Removes features that are highly correlated with the sensitive attribute.
        Note: For this method, the vector s (protected attribute) is assumed to be
        categorical.

        Parameters
        ----------
        correlation_threshold : float, optional
            Features with a correlation value higher than this thresold are
            removed. If None, the top_k parameter is used to determine how many
            features to remove. Defaults to None.
        """
        self.logger = create_logger("methods.preprocessing.Unawareness")
        self.logger.info("Instantiating an Unawareness preprocessing method.")
        self.used_in_inference = True

        self.correlation_threshold = correlation_threshold

    def _correlation_ratio(
        self, categorical_feature: np.ndarray, numeric_feature: np.ndarray
    ):
        """This is a measure of the correlation between a categorical column and
        a numeric column. It measures the variance of the mean of the numeric
        column across different categories of the categorical column. It can
        take values between 0 and 1. A value of 1 indicates that the variance in
        the numeric data is purely due to the difference within the categorical
        data. A value of 0 indicates that the variance in the numeric data is
        completely unaffected by any differences within the categorical data.

        Parameters
        ----------
        categorical_feature : numpy.ndarray
            Categorical column.
        numeric_feature : numpy.ndarray
            Numeric column.

        Returns
        -------
        float
            Correlation ratio value.
        """
        cats, freqs = np.unique(categorical_feature, return_counts=True)
        numeric_mean = np.mean(numeric_feature)
        sig_y_bar = 0
        for i in range(len(cats)):
            category_mean = np.mean(numeric_feature[categorical_feature == cats[i]])
            sig_y_bar += np.square(category_mean - numeric_mean) * freqs[i]
        sig_y = np.sum(np.square(numeric_feature - numeric_mean))
        statistic = np.sqrt(0 if sig_y == 0 else sig_y_bar / sig_y)
        return statistic

    def _cramerv(self, a: np.ndarray, b: np.ndarray):
        """This is a measure of the correlation between two categorical columns.
        Based on the chi-squared metric, the Cramer's V statistic “scales” the
        chi-squared to be a percentage of its maximum possible variation. It can
        take values between 0 and 1, with 1 indicating a complete association
        between the two variables, and a 0 indicating no association. The
        Cramer's V is a heavily biased estimator and tends to overestimate the
        strength of the correlation. Therefore, a biased correction is normally
        applied to the statistic.

        Parameters
        ----------
        a : numpy.ndarray
            First categorical column.
        b : numpy.ndarray
            Second categorical column.

        Returns
        -------
        float
            Cramer's V statistic value.
        """
        contingency = pd.crosstab(index=[a], columns=[b])
        chi2 = chi2_contingency(contingency)[0]
        n = np.sum(contingency.values)
        r, k = contingency.shape
        phi2 = chi2 / n

        phi2_corrected = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_corrected = r - (r - 1) ** 2 / (n - 1)
        k_corrected = k - (k - 1) ** 2 / (n - 1)

        statistic = np.sqrt(phi2_corrected / min(r_corrected - 1, k_corrected - 1))
        return statistic

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        """Calculates correlation between each feature and the sensitive attribute.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Label vector.
        s : pandas.Series
            Protected attribute vector.
        """
        super().fit(X, y, s)

        self.logger.info(
            "Identifying features correlated with the sensitive attribute."
        )

        scores = pd.Series(index=X.columns)

        for col in X.columns:
            if X[col].dtype.name == "category":
                scores[col] = self._cramerv(s.values, X[col].values)
            else:
                scores[col] = self._correlation_ratio(s.values, X[col].values)

        scores = scores.sort_values(ascending=False)
        self.remove_features = list(
            scores.loc[scores >= self.correlation_threshold].index
        )

    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Removes the most correlated features with the sensitive attribute.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Label vector.
        s : pd.Series, optional
            Protected attribute vector.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            The transformed input, X, y, and s.
        """
        super().transform(X, y, s)

        self.logger.info(
            f"Removing most correlated features with sensitive attribute: "
            f"{self.remove_features}"
        )

        X_transformed = X.drop(columns=self.remove_features)

        return X_transformed, y, s
