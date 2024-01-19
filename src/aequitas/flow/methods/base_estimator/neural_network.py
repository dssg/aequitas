from typing import Optional

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils import create_logger, import_object
from .base_estimator import BaseEstimator


class NeuralNetwork(BaseEstimator):
    def __init__(
        self,
        optimizer: str,
        loss: str,
        activation: str,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        **kwargs,
    ):
        """Creates a neural network model.

        Parameters
        ----------
        optimizer : str
            The optimizer to use. Must be a class from the torch.optim package.
        loss_class : str
            The loss function to use. Must be a class from the torch.nn package.
        activation_class : str
            The activation function to use. Must be a class from the torch.nn package.
        learning_rate : float
            The learning rate for the optimizer.
        weight_decay : float
            The weight decay for the optimizer.
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            The batch size to use during training.
        kwargs : dict, optional
            A dictionary containing the hyperparameters for the neural network. Every
            parameter will be passed down to the neural network.
        """
        super().__init__(**kwargs)
        kwargs.pop("unawareness", None)
        self.logger = create_logger("methods.inprocessing.NeuralNetwork")
        self.optimizer_class = optimizer
        self.loss_class = loss
        kwargs["activation_type"] = import_object(activation)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger.debug(f"Instantiating neural network with parameters: {kwargs}.")
        self.model = TorchNeuralNetwork(**kwargs)
        self.optimizer: optim.Optimizer = None

    def fit(self, X: pd.DataFrame, y: pd.Series, s: Optional[pd.Series] = None) -> None:
        """Fits the neural network to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.
        y : pd.Series
            The training labels.
        s : pd.Series, optional
            The sensitive attribute.
        """
        super().fit(X=X, y=y, s=s)
        data = [[X.values[i], y.values[i]] for i in range(X.shape[0])]
        loss_function = import_object(self.loss_class)()
        loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.model._create_architecture(X.shape[1])
        self.logger.debug(f"Instantiating optimizer {self.optimizer_class}.")
        self.optimizer = import_object(self.optimizer_class)(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for _ in tqdm(range(self.epochs)):
            self.model.train()

            # Divide the dataset in batches of the specified size.
            for X_batch, y_batch in tqdm(loader):
                y_pred = self.model(X_batch.float()).cpu()
                loss = loss_function(y_pred.t()[0], y_batch.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(
        self, X: pd.DataFrame, s: Optional[pd.Series] = None
    ) -> pd.Series:
        """Predicts the probability of the positive class.

        Parameters
        ----------
        X : pd.DataFrame
            The data to make predictions on.
        s : pd.Series, optional
            The sensitive attribute.

        Returns
        -------
        pd.Series
            The predicted probabilities.
        """
        super.predict_proba(X=X, s=s)
        self.model.eval()
        tensor_x = torch.tensor(X.values).float()
        preds = self.model(tensor_x)
        return pd.Series(
            preds.detach().to(torch.device("cpu")).numpy().flatten(),
            name="predictions",
            index=X.index,
        )


class TorchNeuralNetwork(nn.Module):
    def __init__(
        self,
        dim_layers: tuple[int],
        activation_type: nn.Module,
        dropout_rate: float,
        batch_norm: bool = True,
        seed: int = 42,
        device: str = "cuda:1",
    ) -> None:
        """Instantiates a Torch neural network.

        Parameters
        ----------
        dim_layers : tuple[int]
            Tuple of integers representing the dimensions of each hidden layer.
        activation_type : nn.Module
            Torch activation function module to be used in the hidden layers.
        dropout_rate : float
            Dropout rate for dropout regularization in hidden layers.
        batch_norm : bool, optional
            Whether to apply batch normalization to the hidden layers. Default is True.
        seed : int, optional
            Random seed for weight initialization. Default is 42.
        device : str, optional
            Device to use for training. Default is "cuda:1".
        """
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.seed = seed
        self.layers = nn.ModuleList()

        self.dim_layers = dim_layers
        self.activation_type = activation_type()
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

    def _create_architecture(self, input_size: int) -> None:
        """Creates the architecture of the neural network.

        Parameters
        ----------
        input_size : int
            The input size of the neural network.
        """
        for size in self.dim_layers:
            if self.dropout_rate:
                self.layers.append(nn.Dropout(self.dropout_rate))
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # update input size for next layer
            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(input_size))
            self.layers.append(self.activation_type)

        self.layers.append(nn.Linear(input_size, 1))
        self.layers.append(nn.Sigmoid())

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes the weights of the neural network.

        Parameters
        ----------
        module : nn.Module
            The neural network module.
        """
        torch.manual_seed(self.seed)
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=1.0)
                module.bias.data.zero_()

    def forward(self, x: torch.tensor):
        """Forward pass of the neural network.

        Parameters
        ----------
        x : torch.tensor
            The input tensor.
        """
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
        return x
