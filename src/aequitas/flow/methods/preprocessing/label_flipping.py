from .preprocessing import PreProcessing

from ...utils import create_logger
from ...utils.imports import import_object

import inspect
import pandas as pd
import math
from typing import Optional, Tuple, Literal, Union, Callable
import numpy as np
from sklearn.ensemble import BaggingClassifier

METHODS = ["ensemble_margin", "residuals"]


class LabelFlipping(PreProcessing):
    def __init__(
        self,
        max_flip_rate: float = 0.1,
        disparity_target: Optional[float] = 0.05,
        score_threshold: Optional[float] = None,
        bagging_max_samples: float = 0.5,
        bagging_base_estimator: Union[
            str, Callable
        ] = "sklearn.tree.DecisionTreeClassifier",
        bagging_n_estimators: int = 10,
        fair_ordering: bool = True,
        ordering_method: Literal["ensemble_margin", "residuals"] = "ensemble_margin",
        unawareness_features: Optional[list] = None,
        seed: int = 42,
        **base_estimator_args,
    ):
        """Flips the labels of a fraction of the training data according to the Fair
        Ordering-Based Noise Correction method.

        Parameters
        ----------
        max_flip_rate : float, optional
            Maximum fraction of the training data to flip, by default 0.1
        disparity_target : float, optional
            The target disparity between the groups (difference between the prevalence
            of a group and the mean prevalence). By default None, which means the
            method will attempt to equalize the prevalence of the groups.
        score_threshold : float, optional
            The threshold above which the labels are flipped. By default None,
            which means the method will flip the labels of the instances with
            a score value higher than 0.
        bagging_max_samples : float, optional
            The number of samples to draw from X to train each base estimator of the
            bagging classifier (with replacement).
        bagging_base_estimator : str, optional
            The base estimator to fit on random subsets of the dataset. By default, the
            base estimator is the sklearn implementation of a decision tree.
        bagging_n_estimators : int, optional
            The number of base estimators in the ensemble, by default 10.
        fair_ordering : bool, optional
            Whether to take additional fairness criteria into account when flipping
            labels, only modifying the labels that contribute to equalizing the
            prevalence of the groups. By default True.
        ordering_method : str, optional
            The method used to calculate the margin of the base estimator. If
            "ensemble_margin", calculates the ensemble margins based on the binary
            predictions of the classifiers. If "residuals", orders the misclassified
            instances based on the average residuals of the classifiers predictions. By
            default "ensemble_margin".
        unawareness_features : list, optional
            The sensitive attributes (or proxies) to ignore when fitting the ensemble
            to enable fairness through unawareness.
        seed : int, optional
            The seed to use when fitting the ensemble.
        **base_estimator_args
            Additional arguments to instantiate the base estimator.

        Examples
        --------
        >>> from aequitas.preprocessing import LabelFlipping
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                n_redundant=0, random_state=42)
        >>> lf = LabelFlipping(bagging_base_estimator=DecisionTreeClassifier,
                max_flip_rate=0.1, max_depth=3)
        >>> lf.fit(X, y)
        >>> X_transformed, y_transformed = lf.transform(X, y)
        """
        self.logger = create_logger("methods.preprocessing.LabelFlipping")
        self.logger.info("Instantiating a LabelFlipping preprocessing method.")

        self.max_flip_rate = max_flip_rate

        if disparity_target is not None:
            if disparity_target < 0 or disparity_target > 1:
                raise ValueError("Disparity target must be a value between 0 and 1.")
            self.disparity_target = disparity_target
        else:
            self.disparity_target = 0

        if score_threshold is not None:
            if score_threshold < 0 or score_threshold > 1:
                raise ValueError("Score threshold must be a value between 0 and 1.")
            self.score_threshold = score_threshold
        else:
            self.score_threshold = 0

        self.bagging_max_samples = bagging_max_samples

        if isinstance(bagging_base_estimator, str):
            bagging_base_estimator = import_object(bagging_base_estimator)
        signature = inspect.signature(bagging_base_estimator)
        if (
            signature.parameters[list(signature.parameters.keys())[-1]].kind
            == inspect.Parameter.VAR_KEYWORD
        ):
            args = (
                base_estimator_args  # Estimator takes **kwargs, so all args are valid
            )
        else:
            args = {
                arg: value
                for arg, value in base_estimator_args.items()
                if arg in signature.parameters
            }
        self.bagging_base_estimator = bagging_base_estimator(**args)
        self.logger.info(
            f"Created base estimator {self.bagging_base_estimator} with params {args}, "
            f"discarded args:{list(set(base_estimator_args.keys()) - set(args.keys()))}"
        )
        self.bagging_n_estimators = bagging_n_estimators

        self.fair_ordering = fair_ordering
        if ordering_method not in METHODS:
            raise ValueError(f"Invalid margin method. Try one of {METHODS}.")
        self.ordering_method = ordering_method
        self.unawareness_features = unawareness_features
        self.used_in_inference = False
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        """
        Fits a bagging classifier to the data. The estimators' can then be used to
        make predictions and calculate the scores to order the instances by.
        If there are categorical features, they are one-hot encoded.

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

        self.logger.info("Fitting LabelFlipping.")

        X_transformed = X.copy()
        if self.unawareness_features is not None:
            X_transformed = X_transformed.drop(columns=self.unawareness_features)

        X_transformed = pd.get_dummies(X_transformed)

        self.ensemble = BaggingClassifier(
            estimator=self.bagging_base_estimator,
            n_estimators=self.bagging_n_estimators,
            max_samples=self.bagging_max_samples,
            random_state=self.seed,
        ).fit(X_transformed, y)

    def _score_instances(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Scores the instances based on the predictions of the ensemble of classifiers.

        If the ordering method is "ensemble_margin", the scores are the ensemble
        margins. If the ordering method is "residuals", the scores are the average
        residuals of the classifiers predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Label vector.

        Returns
        -------
        scores : pd.Series
            The scores of the instances.
        """
        if self.ordering_method == "ensemble_margin":
            y_pred = np.array(
                [clf.predict(X.values) for clf in self.ensemble.estimators_]
            )
            v_1 = y_pred.sum(axis=0)
            v_0 = self.bagging_n_estimators - v_1
            scores = pd.Series(
                np.where(
                    y == 1,
                    (v_1 - v_0) / self.bagging_n_estimators,
                    (v_0 - v_1) / self.bagging_n_estimators,
                ),
                index=X.index,
            )
        elif self.ordering_method == "residuals":
            y_pred = np.array(
                [
                    abs(y - clf.predict_proba(X.values)[:, 1])
                    for clf in self.ensemble.estimators_
                ]
            )
            scores = pd.Series(
                y_pred.sum(axis=0) / self.bagging_n_estimators, index=X.index
            )

        return scores

    def _calculate_group_flips(self, y: pd.Series, s: pd.Series):
        prevalence = y.mean()
        group_prevalences = y.groupby(s).mean()

        min_prevalence = prevalence - self.disparity_target * prevalence
        max_prevalence = prevalence + self.disparity_target * prevalence

        group_flips = {
            group: math.ceil(min_prevalence * len(y[s == group])) - y[s == group].sum()
            if group_prevalences[group] < min_prevalence
            else math.floor(max_prevalence * len(y[s == group])) - y[s == group].sum()
            for group in group_prevalences.index
        }

        return group_flips

    def _label_flipping(self, y: pd.Series, s: Optional[pd.Series], scores: pd.Series):
        """Flips the labels of the desired fraction of the training data.

        If fair_ordering is True, only flips the labels of the instances that contribute
        to equalizing the prevalence of the groups.
        Otherwise, the labels of the instances with the largest score values are
        flipped.

        Parameters
        ----------
        y : pd.Series
            Label vector.
        s : pd.Series, optional
            Protected attribute vector.
        scores : pd.Series
            The scores of the instances.

        Returns
        -------
        y_flipped : pd.Series
            The transformed label vector.
        """
        y_flipped = y.reindex(
            scores.sort_values(
                ascending=(self.ordering_method == "ensemble_margin")
            ).index
        )
        n_flip = int(self.max_flip_rate * len(y))

        if self.fair_ordering:
            group_flips = self._calculate_group_flips(y_flipped, s)
            flip_index = (
                y_flipped.index
                if self.ordering_method == "residuals"
                else y_flipped.loc[scores <= 0].index
            )
            flip_count = 0

            for i in flip_index:
                if abs(scores.loc[i]) < self.score_threshold:
                    break

                if (group_flips[s.loc[i]] > 0 and y.loc[i] == 0) or (
                    group_flips[s.loc[i]] < 0 and y.loc[i] == 1
                ):
                    y_flipped.loc[i] = 1 - y.loc[i]
                    flip_count += 1
                    if group_flips[s.loc[i]] > 0:
                        group_flips[s.loc[i]] -= 1
                    else:
                        group_flips[s.loc[i]] += 1

                if flip_count == n_flip:
                    break

            self.logger.info(f"Flipped {flip_count} instances.")

        else:
            n_above_threshold = scores.loc[abs(scores) >= self.score_threshold].shape[0]
            y_flipped[: min(n_flip, n_above_threshold)] = (
                1 - y_flipped[: min(n_flip, n_above_threshold)]
            )

            self.logger.info(f"Flipped {n_flip} instances.")

        return y_flipped.reindex(y.index)

    def transform(
        self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Flips the labels the specified fraction of the training data according to the
        defined method.

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
        
        self.logger.info("Transforming data with LabelFlipping.")

        if s is None and self.fair_ordering:
            raise ValueError(
                "Sensitive Attribute `s` not passed. Must be passed if `fair_ordering` "
                "is True."
            )

        X_transformed = X.copy()
        if self.unawareness_features is not None:
            X_transformed = X_transformed.drop(columns=self.unawareness_features)

        X_transformed = pd.get_dummies(X_transformed)

        scores = self._score_instances(X_transformed, y)
        y_flipped = self._label_flipping(y, s, scores)

        self.logger.info("Data transformed.")
        return X, y_flipped, s
