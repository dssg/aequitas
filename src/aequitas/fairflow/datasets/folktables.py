import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import requests
from validators import url

from ..utils import LabeledFrame, create_logger
from .dataset import Dataset

VARIANTS = [
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSPublicCoverage",
    "ACSTravelTime",
    "ACSIncome (Sample)",
]

TARGET_FEATURES = {
    "ACSIncome": "PINCP",
    "ACSIncome (Sample)": "PINCP",
    "ACSEmployment": "ESR",
    "ACSMobility": "MIG",
    "ACSPublicCoverage": "PUBCOV",
    "ACSTravelTime": "JWMNP",
}

SENSITIVE_FEATURE = "RAC1P"

CATEGORICAL_FEATURES = []  # To do later

SPLIT_TYPES = ["predefined", "random"]  # Add more if wanted.
SPLIT_VALUES = ["train", "validation", "test"]

DEFAULT_SPLIT = None

DEFAULT_PATH = (Path(__file__).parent / "../../datasets/FolkTables").resolve()

DEFAULT_URL = (
    "https://raw.githubusercontent.com//dssg/aequitas/fairflow-release/"
    "datasets/FolkTables/"
)


class FolkTables(Dataset):
    def __init__(
        self,
        variant: str,
        split_type: str = "predefined",
        splits: dict[str, list[Any]] = DEFAULT_SPLIT,
        path: Union[str, Path] = DEFAULT_PATH,
        seed: int = 42,
        extension: str = "parquet",
        target_feature: Optional[str] = None,
        sensitive_feature: Optional[str] = None,
    ):
        """Instantiate a FolkTables dataset.

        Parameters
        ----------
        variant : str
            The variant of the dataset to load.
        split_type : str, optional
            The type of data split to use. Defaults to "month".
        splits : dict[str, list[Any]], optional
            The proportions of data to use for each split. Defaults to DEFAULT_SPLIT.
        path : Path, optional
            The path to the dataset directory. Defaults to aequitas/datasets/FolkTables.
        seed : int, optional
            Sampling seed for the dataset. Only required in "split_type" == "random".
            Defaults to 42.
        extension : str, optional
            Extension type of the dataset files. Defaults to "parquet".
        """
        super().__init__()

        self.logger = create_logger("datasets.FolkTables")
        self.logger.info("Instantiating a FolkTables dataset.")

        # Validate inputs:
        if variant not in VARIANTS:
            raise ValueError(f"Invalid variant value. Try one of: {VARIANTS}")
        else:
            self.variant = variant
            self.logger.debug(f"Variant: {self.variant}")
        if url(path) or path.exists():
            self._download = False
        else:
            # Download if path does not exist and data not in path
            self._download = True
        self.path = path

        if split_type not in SPLIT_TYPES:
            raise ValueError(f"Invalid split_type value. Try one of: {SPLIT_TYPES}")
        else:
            self.split_type = split_type
            self.splits = splits
            self._validate_splits()
            self.logger.debug("Splits successfully validated.")
        self.extension = extension
        self.seed = seed
        self._data: LabeledFrame = None
        self._train: LabeledFrame = None
        self._validation: LabeledFrame = None
        self._test: LabeledFrame = None
        self.target_feature = (
            TARGET_FEATURES[self.variant] if target_feature is None else target_feature
        )
        self.sensitive_feature = (
            SENSITIVE_FEATURE if sensitive_feature is None else sensitive_feature
        )
        self._indexes = None  # Store indexes of predefined splits

    def _validate_splits(self) -> None:
        """Validate the data splits and raise an error if they are invalid.

        Raises
        ------
        ValueError
            If the splits are missing a required key, the sum of split  proportions is
            invalid, or the month values are invalid.
        """
        if self.split_type == "predefined":
            return
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

    def load_data(self):
        """Load the defined FolkTables dataset."""
        self.logger.info("Loading data.")
        if self._download:
            self._download_data()

        if self.split_type == "predefined":
            path = []
            for split in ["train", "validation", "test"]:
                if isinstance(self.path, str):
                    path.append(self.path + f"/{self.variant}.{split}.{self.extension}")
                else:
                    path.append(self.path / f"{self.variant}.{split}.{self.extension}")
        else:
            path = self.path / f"{self.variant}.{self.extension}"

        if self.extension == "parquet":
            if self.split_type == "predefined":
                datasets = [pd.read_parquet(p) for p in path]
                self._indexes = [d.index for d in datasets]
                self.data = pd.concat(datasets)
            else:
                self.data = pd.read_parquet(path)
        else:
            if self.split_type == "predefined":
                datasets = [pd.read_csv(p) for p in path]
                self._indexes = [d.index for d in datasets]
                self.data = pd.concat(datasets)
            else:
                self.data = pd.read_csv(path)
        for col in CATEGORICAL_FEATURES:
            self.data[col] = self.data[col].astype("category")
        self.logger.info("Loaded data successfully.")
        self.logger.debug("Data shape: {self.data.shape}.")
        self.logger.info("Data loaded successfully.")

    def create_splits(self) -> None:
        """Create train, validation, and test splits from the FolkTables dataset."""
        self.logger.info("Creating data splits.")
        if self.split_type == "random":
            remainder_df = self.data.copy()
            original_size = remainder_df.shape[0]
            for key, value in self.splits.items():
                adjusted_frac = (original_size / remainder_df.shape[0]) * value
                sample = remainder_df.sample(frac=adjusted_frac, random_state=self.seed)
                setattr(self, key, sample)
                sample_indexes = sample.index
                remainder_df = remainder_df.drop(sample_indexes)

        elif self.split_type == "predefined":
            for key, value in zip(["train", "validation", "test"], self._indexes):
                setattr(self, key, self.data.loc[value])
        self.logger.info("Data splits created successfully.")

    def _download_data(self) -> None:
        """Obtains the data from Aequitas repository."""
        self.logger.info("Downloading folktables data from repository.")
        for split in ["train", "validation", "test"]:
            check_path = Path(self.path) / f"{self.variant}.{split}.{self.extension}"
            if not check_path.exists():
                dataset_url = DEFAULT_URL + f"{self.variant}.{split}.{self.extension}"
                self.logger.debug(f"Downloading from {dataset_url}.")
                r = requests.get(dataset_url)
                os.makedirs(check_path.parent, exist_ok=True)
                with open(check_path, "wb") as f:
                    f.write(r.content)
        self.logger.info("Downloaded data successfully.")
