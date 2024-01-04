from typing import Optional

import pandas as pd
from fairgbm import FairGBMClassifier

from ...utils import create_logger
from .inprocessing import InProcessing


class FairGBM(InProcessing):
    def __init__(self, **kwargs):
        """Creates a FairGBM model.

        The FairGBM model is a novel method of InProcessing, where a boosting trees
        algorithm (LightGBM) is subject to pre-defined fairness constraints.

        Parameters
        ----------
        **kwargs : dict, optional
            A dictionary containing the hyperparameters for the FairGBM model.
        """
        self.logger = create_logger("methods.inprocessing.FairGBM")
        self.model = FairGBMClassifier(**kwargs)
        self.logger.debug(f"Instantiating FairGBM with following kwargs: {kwargs}")

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """Fit the FairGBM model to the training data.

        Parameters
        ----------
        X : pandas.DataFrame
            The training input samples.
        y : pandas.Series
            The target values.
        s : pandas.Series, optional
            The sensitive attribute values.
        """
        self.logger.debug(
            f"Input size for model training: {X.shape[0]} rows, "
            f"{X.shape[1]} columns."
        )
        self.model.fit(X=X, y=y, constraint_group=s)

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
        self.logger.debug(f"Input size for model prediction: {X.shape[0]} rows")
        return pd.Series(
            data=self.model.predict_proba(X=X)[:, 1], name="predictions", index=X.index
        )
