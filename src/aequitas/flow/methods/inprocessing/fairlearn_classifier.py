from copy import deepcopy
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions._grid_search._grid_generator import _GridGenerator

from ...utils import create_logger, import_object
from .inprocessing import InProcessing

DEFAULT_GRID_SIZE = 10


class FairlearnClassifier(InProcessing):
    def __init__(
        self,
        reduction: Union[str, Union[object, Callable]],
        estimator: Union[str, Union[object, Callable]],
        constraint: Union[str, Union[object, Callable]],
        **kwargs,
    ):
        """Creates a model from the Fairlearn package.

        Especially designed for the ExponentiatedGradient and GridSearch methods.

        Parameters
        ----------
        reduction : Union[str, callable]
            Reductions method. Either Exponentiated Gradient or Grid Search method. If
            string, it is imported during runtime.
        estimator : Union[str, callable]
            Base estimator for the reductions method. If string, it is imported during
            runtime.
        constraint: Union[str, Moment]
            Constraint for the reductions method. Must be a Moment from the Fairlearn
            package.
        **kwargs : dict, optional
            A dictionary containing the hyperparameters for the reduction method, base
            estimator and constraint. Parameters for the base estimator should be
            included with the prefix `model__`, and for the constraint with the prefix
            `constraint__`. Every other parameter will be passed down to the reductions
            method.
        """
        self.logger = create_logger("methods.inprocessing.ExponentiatedGradient")
        # Importing any object that was passed down as string.
        if isinstance(reduction, str):
            self.logger.debug(f"Importing reduction: '{reduction}'.")
            reduction = import_object(reduction)
        if isinstance(estimator, str):
            self.logger.debug(f"Importing estimator: '{estimator}'.")
            estimator = import_object(estimator)
        if isinstance(constraint, str):
            self.logger.debug(f"Importing constraint: '{constraint}'.")
            constraint = import_object(constraint)

        # Parse keyword arguments
        self.model_kwargs, self.constraint_kwargs, self.kwargs = self.parse_kwargs(
            kwargs
        )

        # Instantiate objects for the method
        self.logger.info(
            f"Instantiating estimator '{estimator}' with parameters:"
            f" {self.model_kwargs}."
        )
        self.estimator = estimator(**self.model_kwargs)

        self.logger.info(
            f"Instantiating constraint '{constraint}' with "
            f"parameters: {self.constraint_kwargs}."
        )
        self.constraint = constraint(**self.constraint_kwargs)

        self.logger.info(
            f"Instantiating reduction '{reduction}' with parameters:" f" {self.kwargs}."
        )
        self.reduction = reduction(
            estimator=self.estimator,
            constraints=self.constraint,
            **self.kwargs,
        )

        # Depending on reduction, use probability mass function method.
        if isinstance(self.reduction, ExponentiatedGradient):
            self.predict_proba_method = lambda clf: clf._pmf_predict
        else:
            self.predict_proba_method = lambda clf: clf.predict_proba

    def fit(self, X: pd.DataFrame, y: pd.Series, s: pd.Series):
        """Fits the fairlearn classifier to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            The target values.
        s : pd.Series
            The protected attribute.
        """
        if isinstance(self.reduction, GridSearch):
            self._generate_grid(X, y, s)
        return self.reduction.fit(X, y, sensitive_features=s)

    def predict_proba(
        self,
        X: pd.DataFrame,
        s: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Use the machine learning model to make predictions on new data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        s : Optional[pd.Series], optional
            The protected attribute.
        """
        return pd.Series(
            data=self.predict_proba_method(self.reduction)(X)[:, 1],
            name="predictions",
            index=X.index,
        )  # Note: This is based on assumption of using LightGBM as base estimator

    def _generate_grid(self, X: pd.DataFrame, y: pd.Series, s: pd.Series) -> None:
        """Generates a grid to pass to GridSearch method.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            The target values.
        s : pd.Series
            The protected attribute.
        """
        dummy_constraint = deepcopy(self.constraint)
        dummy_constraint.load_data(X, y, sensitive_features=s)

        # Randomly select a set of Lagrangian multipliers from the generated grid
        grid = _GridGenerator(
            grid_size=self.kwargs.pop("grid_size", 50),
            grid_limit=self.kwargs.pop("grid_limit", 3.0),
            pos_basis=self.kwargs.pop("pos_basis", dummy_constraint.pos_basis),
            neg_basis=self.kwargs.pop("neg_basis", dummy_constraint.neg_basis),
            neg_allowed=self.kwargs.pop(
                "neg_allowed", dummy_constraint.neg_basis_present
            ),
            force_L1_norm=self.kwargs.pop(
                "force_L1_norm",
                dummy_constraint.default_objective_lambda_vec is not None,
            ),
            grid_offset=None,
        ).grid

        self.rng = np.random.RandomState(self.kwargs.pop("random_state", 42))
        rng_indices = self.rng.choice(grid.shape[1], 2, replace=False)
        grid = grid.iloc[:, rng_indices]

        self.reduction.grid = grid

    @staticmethod
    def parse_kwargs(kwargs: dict[str, Any]) -> tuple[dict, dict, dict]:
        """Parses the keyword arguments for the FairlearnClassifier.

        Parameters
        ----------
        kwargs : dict[str, Any]
            A dictionary containing the hyperparameters for the reduction method, base
            estimator and constraint. Parameters for the base estimator should be
            included with the prefix `model__`, and for the constraint with the prefix
            `constraint__`. Every other parameter will be passed down to the reductions
            method.

        Returns
        -------
        tuple[dict, dict, dict]
            The parameters for the base estimator, constraint, and reductions method, in
            this order.
        """
        MODEL_PREFIX = "model__"
        CONSTRAINT_PREFIX = "constraint__"

        # kwargs for base estimator (or model)
        model_kwargs = {
            k[len(MODEL_PREFIX) :]: v
            for k, v in kwargs.items()
            if k.startswith(MODEL_PREFIX)
        }

        # -> for the constraint
        constraint_kwargs = {
            k[len(CONSTRAINT_PREFIX) :]: v
            for k, v in kwargs.items()
            if k.startswith(CONSTRAINT_PREFIX)
        }

        # -> finally, everything left is a kwarg to the reductions method
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if not any([k.startswith(MODEL_PREFIX), k.startswith(CONSTRAINT_PREFIX)])
        }

        return model_kwargs, constraint_kwargs, kwargs
