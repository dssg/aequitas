from pathlib import Path
from typing import Union
from validators import url

import pandas as pd


class GenericDataset:
    def __init__(
        self,
        train_path: Union[str, Path],
        validation_path: Union[str, Path],
        test_path: Union[str, Path],
        categorical_columns: list[str],
        target_column: str,
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
        self.target_column = target_column
        self.categorical_columns = categorical_columns

        paths = [train_path, validation_path, test_path]
        self.paths = []
        for path in paths:
            # Check if it is an URL or a directory
            if url(path) or Path(path).exists():
                self.paths.append(path)
            else:
                raise ValueError(f"Invalid path: {path}")

        self.extension = extension

    def load_data(self) -> None:
        """Load the dataset from disk."""
        if self.extension == "parquet":
            train = pd.read_parquet(self.paths[0])
            validation = pd.read_parquet(self.paths[1])
            test = pd.read_parquet(self.paths[2])
        elif self.extension == "csv":
            train = pd.read_csv(self.paths[0])
            validation = pd.read_csv(self.paths[1])
            test = pd.read_csv(self.paths[2])

        self.data = [train, validation, test]

    def create_splits(self) -> dict[str, pd.DataFrame]:
        """Split the data into train, validation, and test sets."""
        train, validation, test = self.data
        return {"train": train, "validation": validation, "test": test}
