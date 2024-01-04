from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
from validators import url

from ..utils import create_logger
from .dataset import Dataset

SPLIT_VALUES = ["train", "validation", "test"]

DEFAULT_SPLIT = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}


class GenericDataset(Dataset):
    def __init__(
        self,
        target_feature: str,
        sensitive_feature: str,
        categorical_features: list[str] = [],
        dataset_path: Optional[Union[str, Path]] = None,
        train_path: Optional[Union[str, Path]] = None,
        validation_path: Optional[Union[str, Path]] = None,
        test_path: Optional[Union[str, Path]] = None,
        split_type: Literal["random", "feature"] = "random",
        split_values: dict[Literal["train", "validation", "test"], Any] = DEFAULT_SPLIT,
        split_feature: Optional[str] = None,
        extension: str = "parquet",
        seed: int = 42,
    ):
        """Instantiate a generic dataset. Can either be splitted by the user
        or by the object.

        Parameters
        ----------
        train_path : Union[str, Path]
            The path to the training data. May be URL.
        validation_path : Union[str, Path]
            The path to the validation data. May be URL.
        test_path : Union[str, Path]
            The path to the test data. May be URL.
        categorical_features : list[str]
            The names of the categorical columns in the dataset.
        target_column : str
            The name of the target column in the dataset.
        extension : str, optional
            The extension type of the dataset files. Defaults to "parquet".

        Raises
        ------
        ValueErrorLiteral
            If any of the paths are invalid.
        """
        super().__init__()

        self.logger = create_logger("datasets.GenericDataset")
        self.logger.info("Instantiating a Generic dataset.")

        self.target_feature = target_feature
        self.sensitive_feature = sensitive_feature
        self.categorical_features = categorical_features
        if dataset_path:
            # Validate if other paths are None
            if train_path or validation_path or test_path:
                raise ValueError(
                    "If single dataset path is passed, the other paths must be None."
                )
            if url(dataset_path) or Path(dataset_path).exists():
                self.paths = [dataset_path]
                self.split_required = True
        else:
            # Validate if other paths exist
            if not (train_path and validation_path and test_path):
                raise ValueError(
                    "If multiple dataset paths are passed, the single path must be"
                    "`None`."
                )
            paths = [train_path, validation_path, test_path]
            self.paths = []
            for path in paths:
                # Check if it is an URL or a directory
                if url(path) or Path(path).exists():
                    self.paths.append(path)
                else:
                    raise ValueError(f"Invalid path: {path}")
            self.split_required = False

        self.extension = extension
        self._indexes = None
        self.seed = seed
        if self.split_required:
            self.split_type = split_type
            self.splits = split_values
            self.split_feature = split_feature
            self._validate_splits()

    def _validate_splits(self) -> None:
        """Validate the data splits and raise an error if they are invalid.

        Raises
        ------
        ValueError
            If the splits are missing a required key, the sum of split  proportions is
            invalid, or the month values are invalid.
        """
        for key in ["train", "validation", "test"]:
            if key not in self.splits:
                raise ValueError(f"Missing key in passed splits: {key}")
        if self.split_type == "random":
            split_sum = sum(self.splits.values())
            if split_sum > 1:
                raise ValueError(
                    "Invalid split sizes. Make sure the sum of proportions for all the"
                    " datasets is equal to or lower than 1."
                )
            elif split_sum < 1:
                self.logger.warning(f"Using only {split_sum} of the dataset.")
        elif self.split_type == "feature":
            # We can only validate after reading the dataset.
            if self._data is None:
                pass
            else:
                if self.split_feature is None:
                    raise ValueError(
                        "Split feature must be specified when using feature split."
                    )
                if self.split_feature not in self.data.columns:
                    raise ValueError("Split feature must be a column in the dataset.")
                for value in self.splits.values():
                    if value not in self.data[self.split_feature].unique():
                        raise ValueError(
                            "Split values must be present in the split feature."
                        )

    def load_data(self) -> None:
        """Load the dataset."""
        self.logger.info("Loading data.")
        if self.extension == "parquet":
            read_method = pd.read_parquet
        elif self.extension == "csv":
            read_method = pd.read_csv
        if len(self.paths) == 1:
            self.data = read_method(self.paths[0])
            if self.split_type == "feature":
                self._validate_splits()
        else:
            train = read_method(self.paths[0])
            validation = read_method(self.paths[1])
            test = read_method(self.paths[2])
            self._indexes = [train.index, validation.index, test.index]

            self.data = pd.concat([train, validation, test])
        self.logger.info("Data loaded successfully.")

    def create_splits(self) -> None:
        """Split the data into train, validation, and test sets."""
        self.logger.info("Creating data splits.")
        if self.split_required:
            if self.split_type == "random":
                remainder_df = self.data.copy()
                original_size = remainder_df.shape[0]
                for key, value in self.splits.items():
                    adjusted_frac = (original_size / remainder_df.shape[0]) * value
                    sample = remainder_df.sample(
                        frac=adjusted_frac, random_state=self.seed
                    )
                    setattr(self, key, sample)
                    sample_indexes = sample.index
                    remainder_df = remainder_df.drop(sample_indexes)
            elif self.split_type == "feature":
                for key, value in self.splits.items():
                    setattr(
                        self, key, self.data[self.data[self.split_feature].isin(value)]
                    )
        else:
            self.train = self.data.loc[self._indexes[0]]
            self.validation = self.data.loc[self._indexes[1]]
            self.test = self.data.loc[self._indexes[2]]
        self.logger.info("Data splits created successfully.")
