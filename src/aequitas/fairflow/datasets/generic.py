from pathlib import Path
from typing import Union

import pandas as pd
from validators import url

from ..utils import create_logger
from .dataset import Dataset


class GenericDataset(Dataset):
    def __init__(
        self,
        train_path: Union[str, Path],
        validation_path: Union[str, Path],
        test_path: Union[str, Path],
        categorical_features: list[str],
        target_feature: str,
        sensitive_feature: str,
        extension: str = "parquet",
    ):
        """Instantiate a generic dataset.

        Parameters
        ----------
        train_path : Union[str, Path]
            The path to the training data. May be URL.
        validation_path : Union[str, Path]
            The path to the validation data. May be URL.
        test_path : Union[str, Path]
            The path to the test data. May be URL.
        categorical_columns : list[str]
            The names of the categorical columns in the dataset.
        target_column : str
            The name of the target column in the dataset.
        extension : str, optional
            The extension type of the dataset files. Defaults to "parquet".

        Raises
        ------
        ValueError
            If any of the paths are invalid.
        """
        self.logger = create_logger("datasets.GenericDataset")
        self.logger.info("Instantiating a Generic dataset.")

        self.target_feature = target_feature
        self.sensitive_feature = sensitive_feature
        self.categorical_features = categorical_features

        paths = [train_path, validation_path, test_path]
        self.paths = []
        for path in paths:
            # Check if it is an URL or a directory
            if url(path) or Path(path).exists():
                self.paths.append(path)
            else:
                raise ValueError(f"Invalid path: {path}")

        self.extension = extension
        self._indexes = None

    def load_data(self) -> None:
        """Load the dataset from disk."""
        self.logger.info("Loading data.")
        if self.extension == "parquet":
            train = pd.read_parquet(self.paths[0])
            validation = pd.read_parquet(self.paths[1])
            test = pd.read_parquet(self.paths[2])
        elif self.extension == "csv":
            train = pd.read_csv(self.paths[0])
            validation = pd.read_csv(self.paths[1])
            test = pd.read_csv(self.paths[2])
        self._indexes = [train.index, validation.index, test.index]

        self.data = pd.concat([train, validation, test])
        self.logger.info("Data loaded successfully.")

    def create_splits(self) -> None:
        """Split the data into train, validation, and test sets."""
        self.logger.info("Creating data splits.")
        self.train = self.data.loc[self._indexes[0]]
        self.validation = self.data.loc[self._indexes[1]]
        self.test = self.data.loc[self._indexes[2]]
        self.logger.info("Data splits created successfully.")
