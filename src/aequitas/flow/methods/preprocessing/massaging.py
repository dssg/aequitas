from typing import Optional, Union, Callable

import pandas as pd
import math

from ...utils import create_logger
from ...utils.imports import instantiate_object
from .preprocessing import PreProcessing


class Massaging(PreProcessing):
    def __init__(
        self,
        classifier: Union[str, Callable] = "sklearn.naive_bayes.GaussianNB",
        **classifier_args,
    ):
        """
        Instantiates a Massaging preprocessing method.

        Flips selected labels to reduce disparity between groups.
        """
        self.logger = create_logger("methods.preprocessing.Massaging")
        self.logger.info("Instantiating a Massaging preprocessing method.")

        self.classifier = instantiate_object(classifier, **classifier_args)
        self.logger.info(f"Created base estimator {self.classifier}")

    def _rank(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]
    ) -> tuple[list, list]:
        features = pd.concat([X, s], axis=1)
        features = pd.get_dummies(features)
        R = self.classifier.fit(features, y)
        scores = pd.Series(R.predict_proba(features)[:, 1], index=X.index)

        pr = []
        dem = []

        for g in s.unique():
            prevalence = y[s == g].mean()
            if prevalence < y.mean():
                pr += list(X.loc[(s == g) & (y == 0)].index)
            elif prevalence > y.mean():
                dem += list(X.loc[(s == g) & (y == 1)].index)

        pr = scores.loc[pr].sort_values(ascending=False).index
        dem = scores.loc[dem].sort_values(ascending=True).index

        return pr, dem

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        """Fits a classifier to the data and orders the instances by the predictions.
        Promotion candidates are the instances with negative label in the group with
        lowest prevalence and demotion candidates are the instances with positive
        label in the group with highest prevalence. The number of instances to be
        flipped is calculated to equalize the prevalences of the groups.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Label vector.
        s : pandas.Series
            Protected attribute vector.
        """
        self.logger.info("Fitting Massaging preprocessing method.")
        self.pr, self.dem = self._rank(X, y, s)

        g_pr = s.loc[self.pr].unique()
        g_dem = s.loc[self.dem].unique()

        d_b = y.loc[s.isin(g_pr)].mean()
        d_w = y.loc[s.isin(g_dem)].mean()
        d = d_w - d_b

        self.m = math.ceil(
            (d * y.loc[s.isin(g_pr)].shape[0] * y.loc[s.isin(g_dem)].shape[0])
            / y.shape[0]
        )
        self.logger.info("Massaging preprocessing method fitted.")

    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Transforms the data by flipping the calculated number of label of the top
        candidates in the promotion and the demotion groups.

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
        self.logger.info("Transforming data with Massaging preprocessing method.")
        y_corrected = y.copy()
        y_corrected.loc[self.pr[: self.m]] = 1
        y_corrected.loc[self.dem[: self.m]] = 0
        self.logger.info("Data transformed with Massaging preprocessing method.")
        return X, y_corrected, s
