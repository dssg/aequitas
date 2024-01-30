from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ...utils import create_logger
from .preprocessing import PreProcessing


class BackwardFeatureElimination(PreProcessing):
    def __init__(
        self,
        auc_threshold: Optional[int] = 0.5,
        feature_importance_threshold: Optional[float] = 0.1,
        n_estimators: Optional[int] = 10,
        seed: int = 0,
    ):
        """Iterively removes the most important features with respect to the sensitive
        attribute.

        Parameters
        ----------
        auc_threshold : int, optional
            The value of AUC above which the removal of features continues. Defaults to
            0.5.
        feature_importance_threshold : float, optional
            The value of feature importance above which the most important feature needs
            to have to be removed. Defaults to 0.1.
        n_estimators : int, optional
            The number of trees in the random forest. Defaults to 10.
        seed : int, optional
            The seed for the random forest. Defaults to 0.
        """
        self.logger = create_logger("methods.preprocessing.BackwardFeatureElimination")
        self.logger.info(
            "Instantiating a BackwardFeatureElimination preprocessing method."
        )
        self.used_in_inference = True

        self.auc_threshold = auc_threshold
        self.feature_importance_threshold = feature_importance_threshold
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        """Iteratively removes the most important features to predict the sensitive
        attribute.

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

        self.logger.info("Identifying features to remove.")

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed
        )

        features = pd.concat([X, y], axis=1)
        features = pd.get_dummies(features)
        target = s.copy()

        features_train, features_val, target_train, target_val = train_test_split(
            features, target
        )
        self.remove_features = []

        while features_train.shape[1] > 1:
            rf.fit(features_train, target_train)
            predictions = rf.predict_proba(features_val)[:, 1]
            auc = roc_auc_score(target_val, predictions)

            if auc > self.auc_threshold:
                scores = pd.Series(
                    rf.feature_importances_, index=features_train.columns
                )
                feature = scores.sort_values(ascending=False).index[0]
                if scores[feature] < self.feature_importance_threshold:
                    break

                i = feature.rfind("_")
                if feature[:i] in X.columns:
                    eliminate = [
                        col
                        for col in features_train.columns
                        if col.startswith(feature[:i])
                    ]
                    self.remove_features.append(feature[:i])
                else:
                    eliminate = [feature]
                    self.remove_features.append(feature)

                features_train = features_train.drop(columns=eliminate)
                features_val = features_val.drop(columns=eliminate)
            else:
                break

    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Removes the features which are related with the sensitive attribute the most
        from the data.

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
