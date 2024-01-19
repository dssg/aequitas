from typing import Optional

import pandas as pd
from lightgbm import LGBMClassifier

from ...utils import create_logger
from .base_estimator import BaseEstimator


class LightGBM(BaseEstimator):
    def __init__(self, **kwargs):
        """Creates a LightGBM model.

        Parameters
        ----------
        **kwargs : dict, optional
            A dictionary containing the hyperparameters for the FairGBM model.
        """
        super().__init__(**kwargs)
        kwargs.pop("unawareness", None)
        self.logger = create_logger("methods.base_estimator.LightGBM")
        self.logger.info("Instantiating LightGBM model.")
        self.model = LGBMClassifier(**kwargs, verbose=-1)
        self.logger.debug(f"Instantiating LightGBM with following kwargs: {kwargs}")

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """Fit the LightGBM model to the training data.

        Parameters
        ----------
        X : pandas.DataFrame
            The training input samples.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The sensitive attribute values.
        """
        super().fit(X=X, y=y, s=s)
        self.logger.info("Fitting LightGBM model.")
        self.logger.debug(
            f"Input size for model training: {X.shape[0]} rows, "
            f"{X.shape[1]} columns."
        )
        self.model.fit(
            X=X,
            y=y,
        )
        self.logger.info("Finished fitting LightGBM model.")

    def predict_proba(
        self, X: pd.DataFrame, s: Optional[pd.Series] = None
    ) -> pd.Series:
        """Predict class probabilities for the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input samples to predict.
        s : pandas.Series, optional
            The sensitive attribute values.

        Returns
        -------
        pandas.Series
            The predicted class probabilities.
        """
        super().predict_proba(X=X, s=s)
        self.logger.info("Predicting with LightGBM model.")
        self.logger.debug(f"Input size for model prediction: {X.shape[0]} rows")
        preds = pd.Series(
            data=self.model.predict_proba(X=X)[:, 1], name="predictions", index=X.index
        )
        self.logger.info("Finished predicting with LightGBM model.")
        return preds
