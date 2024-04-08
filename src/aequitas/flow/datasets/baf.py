import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from datasets import load_dataset
from validators import url

from ..utils import create_logger
from .dataset import Dataset

VARIANTS = ["Base", "TypeI", "TypeII", "TypeIII", "TypeIV", "TypeV", "Sample"]

CATEGORICAL_COLUMNS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

LABEL_COLUMN = "fraud_bool"

SENSITIVE_COLUMN = "customer_age_bin"

SPLIT_TYPES = ["random", "month"]  # Add more if wanted.
SPLIT_VALUES = ["train", "validation", "test"]

DEFAULT_SPLIT = {
    "train": (0, 1, 2, 3, 4, 5),
    "validation": (6,),
    "test": (7,),
}

DEFAULT_PATH = (Path(__file__).parent / "../../datasets/BankAccountFraud").resolve()

HUGGINGFACE_REPO = "ines-osilva/baf-testing"


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
    label_column : str, optional
        Name of the label column. If None, defaults to "fraud_bool".
    sensitive_column : str, optional
        Name of the sensitive column. If None, defaults to "customer_age_bin".
    include_month : bool, optional
        whether to include the month column in the columns. Defaults to True.
    age_cutoff : int, optional
        Age cutoff for creating the binary age column, if using age as the
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
        label_column: Optional[str] = None,
        sensitive_column: Optional[str] = None,
        include_month: bool = True,
        age_cutoff: int = 50,
    ):
        super().__init__()

        self.label_column = LABEL_COLUMN if label_column is None else label_column

        if sensitive_column == "customer_age" or sensitive_column is None:
            self.sensitive_column = SENSITIVE_COLUMN
        elif sensitive_column not in CATEGORICAL_COLUMNS:
            raise ValueError(
                f"Invalid sensitive column value. Try one of: {CATEGORICAL_COLUMNS}"
            )
        else:
            self.sensitive_column = sensitive_column

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
        for col in CATEGORICAL_COLUMNS:
            self.data[col] = self.data[col].astype("category")

        if self.sensitive_column == "customer_age_bin":
            self.data["customer_age_bin"] = self.data["customer_age"] >= self.age_cutoff
            self.data["customer_age_bin"] = self.data["customer_age_bin"].astype(
                "category"
            )

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
        """Obtains the data of the sample dataset from HuggingFace."""
        self.logger.info("Downloading sample data from HuggingFace.")

        check_path = Path(self.path) / f"{self.variant}.{self.extension}"
        if not check_path.exists():
            self.logger.debug(f"Downloading from {HUGGINGFACE_REPO}.")
            dataset = load_dataset(
                HUGGINGFACE_REPO,
                self.variant,
                token="",
            )["train"].to_pandas()
            os.makedirs(check_path.parent, exist_ok=True)
            if self.extension == "parquet":
                dataset.to_parquet(check_path)
            elif self.extension == "csv":
                dataset.to_csv(check_path, index=False)
        self.logger.info("Downloaded data successfully.")
