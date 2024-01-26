import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import requests
from validators import url

from ..utils import create_logger
from .dataset import Dataset

VARIANTS = ["Base", "TypeI", "TypeII", "TypeIII", "TypeIV", "TypeV", "Sample"]

CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

TARGET_FEATURE = "fraud_bool"

SENSITIVE_FEATURE = "customer_age_bin"

SPLIT_TYPES = ["random", "month"]  # Add more if wanted.
SPLIT_VALUES = ["train", "validation", "test"]

DEFAULT_SPLIT = {
    "train": (0, 1, 2, 3, 4, 5),
    "validation": (6,),
    "test": (7,),
}

DEFAULT_PATH = (Path(__file__).parent / "../../datasets/BankAccountFraud").resolve()

DEFAULT_URL = "https://raw.githubusercontent.com//dssg/aequitas/master/datasets/BankAccountFraud/"


class BankAccountFraud(Dataset):
    """Instantiate a BankAccountFraud dataset.

    Parameters
    ----------
    variant : str
        The variant of the dataset to load.
    split_type : str, optional
        The type of data split to use. Defaults to "month".
    splits : dict[str, list[Any]], optional
        The proportions of data to use for each split. Defaults to original splits.
    path : Path, optional
        The path to the dataset directory. Defaults to ../datasets/BankAccountFraud.
    seed : int, optional
        Sampling seed for the dataset. Only required in "split_type" == "random".
        Defaults to 42.
    extension : str, optional
        Extension type of the dataset files. Defaults to "parquet".
    target_feature : str, optional
        Name of the target feature. If None, defaults to "fraud_bool".
    sensitive_feature : str, optional
        Name of the sensitive feature. If None, defaults to "customer_age_bin".
    include_month : bool, optional
        whether to include the month column in the features. Defaults to True.
    age_cutoff : int, optional
        Age cutoff for creating the binary age feature, if using age as the
        sensitive attribute. Defaults to 50.
    """

    def __init__(
        self,
        variant: str,
        split_type: str = "month",
        splits: dict[str, list[Any]] = DEFAULT_SPLIT,
        path: Union[str, Path] = DEFAULT_PATH,
        seed: int = 42,
        extension: str = "parquet",
        target_feature: Optional[str] = None,
        sensitive_feature: Optional[str] = None,
        include_month: bool = True,
        age_cutoff: int = 50,
    ):
        super().__init__()

        self.target_feature = (
            TARGET_FEATURE if target_feature is None else target_feature
        )

        if sensitive_feature == "customer_age" or sensitive_feature is None:
            self.sensitive_feature = SENSITIVE_FEATURE
        elif sensitive_feature not in CATEGORICAL_FEATURES:
            raise ValueError(
                f"Invalid sensitive feature value. Try one of: {CATEGORICAL_FEATURES}"
            )
        else:
            self.sensitive_feature = sensitive_feature

        self.age_cutoff = age_cutoff

        self.logger = create_logger("datasets.BankAccountFraud")
        self.logger.info("Instantiating a BankAccountFraud dataset.")

        # Validate inputs:
        if variant not in VARIANTS:
            raise ValueError(f"Invalid variant value. Try one of: {VARIANTS}")
        else:
            self.variant = variant
            self.logger.debug(f"Variant: {self.variant}")
        if url(path) or path.exists():
            self.path = path
            self._download = False
        else:
            self.path = path
            self._download = True
        if split_type not in SPLIT_TYPES:
            raise ValueError(f"Invalid split_type value. Try one of: {SPLIT_TYPES}")
        else:
            self.split_type = split_type
            self.splits = splits
            self._validate_splits()
            self.logger.debug("Splits successfully validated.")
        self.extension = extension
        self.seed = seed
        self.data: pd.DataFrame = None
        self.include_month = include_month

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
        elif self.split_type == "month":
            all_vals = [val for val_list in self.splits.values() for val in val_list]
            if len(all_vals) > len(set(all_vals)):
                raise ValueError("Month values are repeated in different sets.")
            if max(all_vals) > 7 or min(all_vals) < 0:
                raise ValueError("Values outside the supported month range.")

    def load_data(self):
        """Load the defined BankAccountFraud dataset."""
        if self._download:
            self._download_data()

        if isinstance(self.path, str):
            path = self.path + f"/{self.variant}.{self.extension}"
        else:
            path = self.path / f"{self.variant}.{self.extension}"
        self.logger.info(f"Loading data from {path}")
        if self.extension == "parquet":
            self.data = pd.read_parquet(path)
        else:
            self.data = pd.read_csv(path)
        for col in CATEGORICAL_FEATURES:
            self.data[col] = self.data[col].astype("category")

        if self.sensitive_feature == "customer_age_bin":
            self.data["customer_age_bin"] = self.data["customer_age"] >= self.age_cutoff
            self.data["customer_age_bin"] = self.data["customer_age_bin"].astype(int)

    def create_splits(self) -> None:
        """Create train, validation, and test splits for the dataset."""
        if self.data is None:
            raise ValueError('Data is not loaded yet. run "BankAccountFraud.load_data"')
        if self.split_type == "random":
            remainder_df = self.data.copy()
            original_size = remainder_df.shape[0]
            for key, value in self.splits.items():
                adjusted_frac = (original_size / remainder_df.shape[0]) * value
                sample = remainder_df.sample(frac=adjusted_frac, random_state=self.seed)
                setattr(self, key, sample)
                sample_indexes = sample.index
                remainder_df = remainder_df.drop(sample_indexes)

        elif self.split_type == "month":
            for key, value in self.splits.items():
                if self.include_month:
                    setattr(self, key, self.data[self.data["month"].isin(value)].copy())
                else:
                    setattr(
                        self,
                        key,
                        self.data[self.data["month"].isin(value)].drop(
                            columns=["month"]
                        ),
                    )

    def _download_data(self) -> None:
        """Obtains the data of the sample dataset from Aequitas repository."""
        self.logger.info("Downloading sample data from repository.")

        check_path = Path(self.path) / f"{self.variant}.{self.extension}"
        if not check_path.exists():
            dataset_url = DEFAULT_URL + f"{self.variant}.{self.extension}"
            self.logger.debug(f"Downloading from {dataset_url}.")
            r = requests.get(dataset_url)
            os.makedirs(check_path.parent, exist_ok=True)
            with open(check_path, "wb") as f:
                f.write(r.content)
        self.logger.info("Downloaded data successfully.")
