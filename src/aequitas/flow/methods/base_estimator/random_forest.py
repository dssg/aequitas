from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from ...utils import create_logger
from .base_estimator import BaseEstimator


class RandomForest(BaseEstimator):
    def __init__(self, **kwargs):
        """Creates a sklearn random forest model.

        Parameters
        ----------
        **kwargs : dict, optional
            A dictionary containing the hyperparameters for the Random Forest model.
        """
        super().__init__(**kwargs)
        kwargs.pop("unawareness", None)
        self.logger = create_logger("methods.base_estimator.RandomForest")
        self.logger.info("Instantiating RandomForest model.")
        self.model = RandomForestClassifier(**kwargs)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.cat_feats = None
        self.logger.debug(f"Instantiating RandomForest with following kwargs: {kwargs}")

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """Fit the RandomForest model to the training data.

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
        self.logger.info("Fitting RandomForest model.")
        self.logger.debug(
            f"Input size for model training: {X.shape[0]} rows, "
            f"{X.shape[1]} columns."
        )
        # For now, we will proceed with one-hot encoding for categorical features
        self.cat_feats = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.logger.debug(f"Found {len(self.cat_feats)} categorical features.")
        if len(self.cat_feats) > 0:
            self.logger.debug(f"Encoding categorical features: {self.cat_feats}")
            X = pd.concat(
                [
                    X.drop(columns=self.cat_feats),
                    pd.DataFrame(
                        self.encoder.fit_transform(X[self.cat_feats]).toarray(),
                        columns=self.encoder.get_feature_names_out(self.cat_feats),
                        index=X.index,
                    ),
                ],
                axis=1,
            )
        self.model.fit(
            X=X,
            y=y,
        )
        self.logger.info("Finished fitting Random Forest model.")

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
        self.logger.info("Predicting with RandomForest model.")
        self.logger.debug(f"Input size for model prediction: {X.shape[0]} rows")
        if len(self.cat_feats) > 0:
            self.logger.debug(f"Encoding categorical features: {self.cat_feats}")
            X = pd.concat(
                [
                    X.drop(columns=self.cat_feats),
                    pd.DataFrame(
                        self.encoder.transform(X[self.cat_feats]).toarray(),
                        columns=self.encoder.get_feature_names_out(self.cat_feats),
                        index=X.index,
                    ),
                ],
                axis=1,
            )
        preds = pd.Series(
            data=self.model.predict_proba(X=X)[:, 1], name="predictions", index=X.index
        )
        self.logger.info("Finished predicting with RandomForest model.")
        return preds
