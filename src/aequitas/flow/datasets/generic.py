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
        label_column: str,
        sensitive_column: str,
        df: Optional[pd.DataFrame] = None,
        categorical_columns: list[str] = [],
        dataset_path: Optional[Union[str, Path]] = None,
        train_path: Optional[Union[str, Path]] = None,
        validation_path: Optional[Union[str, Path]] = None,
        test_path: Optional[Union[str, Path]] = None,
        split_type: Literal["random", "column"] = "random",
        split_values: dict[Literal["train", "validation", "test"], Any] = DEFAULT_SPLIT,
        split_column: Optional[str] = None,
        extension: str = "parquet",
        seed: int = 42,
    ):
        """Instantiate a generic dataset. Can either be splitted by the user
        or by the object.

        Parameters
        ----------
        label_column : str
            The name of the label column in the dataset.
        sensitive_column : str
            The name of the sensitive column in the dataset.
        df : pd.DataFrame, optional
            The dataset to be used. If None, the dataset will be loaded from the
            specified paths. Defaults to None.
        dataset_path : Union[str, Path]
            The path to the dataset. May be URL.
        train_path : Union[str, Path]
            The path to the training data. May be URL.
        validation_path : Union[str, Path]
            The path to the validation data. May be URL.
        test_path : Union[str, Path]
            The path to the test data. May be URL.
        categorical_columns : list[str]
            The names of the categorical columns in the dataset.
        label_column : str
            The name of the label column in the dataset.
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

        self.label_column = label_column
        self.sensitive_column = sensitive_column
        self.categorical_columns = categorical_columns
        if dataset_path:
            # Validate if other paths are None
            if train_path or validation_path or test_path:
                raise ValueError(
                    "If single dataset path is passed, the other paths must be None."
                )
            if url(dataset_path) or Path(dataset_path).exists():
                self.paths = [dataset_path]
                self.split_required = True
        elif df is not None:
            self.data = df
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
            self.split_column = split_column
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
        elif self.split_type == "column":
            # We can only validate after reading the dataset.
            if self._data is None:
                pass
            else:
                if self.split_column is None:
                    raise ValueError(
                        "Split column must be specified when using column split."
                    )
                if self.split_column not in self.data.columns:
                    raise ValueError("Split column must be a column in the dataset.")
                for value in self.splits.values():
                    if value not in self.data[self.split_column].unique():
                        raise ValueError(
                            "Split values must be present in the split column."
                        )

    def load_data(self) -> None:
        """Load the dataset."""
        self.logger.info("Loading data.")
        if self._data is not None:
            return
        if self.extension == "parquet":
            self.logger.info("Loading data from parquet.")
            read_method = pd.read_parquet
        elif self.extension == "csv":
            read_method = pd.read_csv
        if len(self.paths) == 1:
            self.logger.info("Loading data from parquet single file.")
            self.data = read_method(self.paths[0])
            if self.split_type == "column":
                self._validate_splits()
        else:
            train = read_method(self.paths[0])
            train_index = train.index[-1]
            validation = read_method(self.paths[1])
            validation.set_index(validation.index + train_index + 1, inplace=True)
            validation_index = validation.index[-1]
            test = read_method(self.paths[2])
            test.set_index(test.index + validation_index + 1, inplace=True)
            self._indexes = [train.index, validation.index, test.index]

            self.data = pd.concat([train, validation, test]).reset_index(drop=True)
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
                    adjusted_frac = min(adjusted_frac, 1.0)
                    sample = remainder_df.sample(
                        frac=adjusted_frac, random_state=self.seed
                    )
                    setattr(self, key, sample)
                    sample_indexes = sample.index
                    remainder_df = remainder_df.drop(sample_indexes)
            elif self.split_type == "column":
                for key, value in self.splits.items():
                    setattr(
                        self, key, self.data[self.data[self.split_column].isin(value)]
                    )
        else:
            self.train = self.data.loc[self._indexes[0]]
            self.validation = self.data.loc[self._indexes[1]]
            self.test = self.data.loc[self._indexes[2]]
        self.logger.info("Data splits created successfully.")
