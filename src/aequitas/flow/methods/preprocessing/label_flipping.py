from .preprocessing import PreProcessing

from ...utils import create_logger
from ...utils.imports import import_object

import inspect
import pandas as pd
from typing import Optional, Tuple, Literal
import numpy as np
from sklearn.ensemble import BaggingClassifier

METHODS = ["ensemble_margin", "residuals"]

class LabelFlipping(PreProcessing):
    def __init__(
            self,
            flip_rate: float = 0.1,
            bagging_max_samples: float = 0.5,
            base_estimator: str = "sklearn.tree.DecisionTreeClassifier", 
            n_estimators: int = 10,
            fair_ordering: bool = True,
            ordering_method: Literal["ensemble_margin", "residuals"] = "ensemble_margin",
            seed: int = 42,
            **base_estimator_args
        ):
        """Flips the labels of a fraction of the training data according to the Fair 
        Ordering-Based Noise Correction method.
        
        Parameters
        ----------
        flip_rate : float, optional
            Maximum fraction of the training data to flip, by default 0.1
        bagging_max_samples : float, optional
            The number of samples to draw from X to train each base estimator of the 
            bagging classifier (with replacement).
        base_estimator : str, optional
            The base estimator to fit on random subsets of the dataset. By default, the 
            base estimator is the sklearn implementation of a decision tree.
        base_estimator_args : dict, optional    
            Additional arguments to pass to the base estimator.
        n_estimators : int, optional
            The number of base estimators in the ensemble, by default 10.
        fair_ordering : bool, optional  
            Whether to take additional fairness criteria into account when flipping labels,
            only modifying the labels that contribute to equalizing the prevalence of the
            groups. By default True.
        ordering_method : str, optional
            The method used to calculate the margin of the base estimator. If "ensemble_margin",
            calculates the ensemble margins based on the binary predictions of the classifiers.
            If "residuals", oreders the missclafied instances based on the average residuals of the
            classifiers predictions. By default "ensemble_margin".
        """
        self.logger = create_logger("methods.preprocessing.LabelFlipping")
        self.logger.info("Instantiating a LabelFlipping preprocessing method.")

        self.flip_rate = flip_rate
        self.bagging_max_samples = bagging_max_samples

        base_estimator = import_object(base_estimator)
        args = {arg: value for arg, value in base_estimator_args.items() if arg in inspect.signature(base_estimator).parameters}
        self.base_estimator = base_estimator(**args)
        self.logger.info(f'Created base estimator {self.base_estimator} with params {args}, discarded args: {list(set(base_estimator_args.keys()) - set(args.keys()))}')
        self.n_estimators = n_estimators

        self.fair_ordering = fair_ordering
        if ordering_method not in METHODS:
            raise ValueError(f"Invalid margin method. Try one of {METHODS}.")
        self.ordering_method = ordering_method
        self.used_in_inference = False
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> None:
        pass

    def _score_instances(self, X, y, estimators):

        if self.ordering_method == "ensemble_margin":
            scores = pd.Series(dtype=float)
            y_pred = np.array([clf.predict(X.values) for clf in estimators])
            for i in X.index:
                v_1 = y_pred[:,i].sum()
                v_0 = y_pred.shape[0] - v_1
                if y.loc[i] == 1:
                    scores.loc[i] = (v_1 - v_0) / self.n_estimators
                else:    
                    scores.loc[i] = (v_0 - v_1) / self.n_estimators
        
        elif self.ordering_method == "residuals":
            y_pred = np.array([abs(y - clf.predict_proba(X.values)[:,1]) for clf in estimators])
            scores = pd.Series(y_pred.sum(axis=0) / self.n_estimators, index=X.index)

        return scores
    
    def _calculate_prevalence_disparity(self, y: pd.Series, s: pd.Series):
        prevalence_0 = y.loc[s == 0].value_counts()[1] / y.loc[s==0].shape[0]
        prevalence_1 = y.loc[s == 1].value_counts()[1] / y.loc[s==1].shape[0]

        return prevalence_0 - prevalence_1
    
    def _label_flipping(self, y: pd.Series, s: Optional[pd.Series], scores: pd.Series):
        y_flipped = y.reindex(scores.sort_values(ascending=(self.ordering_method == "ensemble_margin")).index)
        n_flip = int(self.flip_rate*len(y))

        if self.fair_ordering: # TO DO: if prevalence disparity equalized/inverts, stop flipping or start iterating the instances that have high margins and weren't flipped?
            
            disparity = self._calculate_prevalence_disparity(y_flipped, s)
            flip_index = y_flipped.index if self.ordering_method == "residuals" else y_flipped.loc[scores <= 0].index
            flip_count = 0

            for i in flip_index:
                if (disparity > 0 and s.loc[i] != y_flipped.loc[i]) or (disparity < 0 and s.loc[i] == y_flipped.loc[i]):
                    y_flipped.loc[i] = 1 - y_flipped.loc[i]
                    disparity = self._calculate_prevalence_disparity(y_flipped, s)
                    flip_count += 1

                if flip_count == n_flip:
                    break

            self.logger.info(f"Flipped {flip_count} instances.")

        else:
            y_flipped[:n_flip] = 1 - y_flipped[:n_flip]

            self.logger.info(f"Flipped {n_flip} instances.")

        return y_flipped.reindex(y.index)

    def transform(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        self.logger.info("Transforming data with LabelFlipping.")

        if s is None and self.fair_ordering:
            raise ValueError("Sensitive Attribute `s` not passed. Must be passed if `fair_ordering` is True.")
        
        X_num = pd.get_dummies(X)

        bagging = BaggingClassifier(estimator=self.base_estimator, 
                                    n_estimators=self.n_estimators, 
                                    max_samples=self.bagging_max_samples,
                                    random_state=self.seed).fit(X_num, y)
        
        scores = self._score_instances(X_num, y, bagging.estimators_)
        y_flipped = self._label_flipping(y, s, scores)

        self.logger.info("Data transformed.")
        return X, y_flipped, s
        
