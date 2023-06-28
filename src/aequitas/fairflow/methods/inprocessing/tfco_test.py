import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import tensorflow_constrained_optimization as tfco

from tensorflow.keras import activations, layers, losses, Sequential, Input
from tensorflow_constrained_optimization.python.rates import loss
from tensorflow import keras

from . import InProcessing

# Create a wrapper class for TensorFlow Constrained Optimization
class TensorflowConstrainedOptimization(InProcessing):
    def__init__(
        self,
        n_jobs,
        batch_size,
        max_epochs,
        input_dim,
        hidden_layers,
        use_batch_norm,
        dropout,
        lr,
        betas,
        weight_decay,
        amsgrad,
        protected_attribute,
        fpr_diff=0.05,
        seed=42,
    ):
        self.model = None
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.protected_attribute = protected_attribute
        self.fpr_diff = fpr_diff
        self.seed = seed
    
    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """Fit the model to the data.
        Args:
            X: The features of the data.
            y: The labels of the data.
            s: The protected attribute of the data.
        """
        # Create a dataset wrapper
        dataset = _DatasetWrapper(X, y, self.protected_attribute)
        # Create a model
        self.model = _create_model(
            self.input_dim,
            self.hidden_layers,
            self.use_batch_norm,
            self.dropout,
            self.lr,
            self.betas,
            self.weight_decay,
            self.amsgrad,
        )
        # Create a loss function
        loss_fn = _create_loss_fn(self.fpr_diff, self.n_jobs)
        # Create a constraint
        constraint = _create_constraint(self.protected_attribute, self.n_jobs)
        # Create a rate
        rate = _create_rate(self.protected_attribute, self.n_jobs)
        # Create a optimizer
        optimizer = _create_optimizer(self.lr, self.betas, self.weight_decay, self.amsgrad)
        # Create a minimization problem
        problem = _create_minimization_problem(
            self.model, loss_fn, constraint, rate, optimizer
        )
        # Create a solver
        solver = _create_solver(problem)
        # Solve the problem
        solver.solve()

    def predict_proba(self, X: pd.DataFrame, s: Optional[pd.Series] = None) -> pd.Series:
        """Predict the probability of the positive class.
        Args:
            X: The features of the data.
            s: The protected attribute of the data.
        Returns:
            The probability of the positive class.
        """
        return pd.Series(self.model.predict(X))

    def _create_model(self):
        