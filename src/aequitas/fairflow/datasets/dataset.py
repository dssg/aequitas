from abc import ABC, abstractmethod

import pandas as pd

from ..utils import LabeledFrame


class Dataset(ABC):
    def __init__(self) -> None:
        self._data = None
        self._train = None
        self._validation = None
        self._test = None
        self.target_feature = None
        self.sensitive_feature = None
        self.categorical_features = None

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset. Populates the `data` property."""
        pass

    @abstractmethod
    def create_splits(self) -> None:
        """
        Create the train, validation and test splits.Populates the `train`, `validation`
        and `test` properties.
        """
        pass

    @property
    def data(self) -> LabeledFrame:
        """Return the dataset."""
        if self._data is not None:
            return self._data
        else:
            raise ValueError("Data is not loaded yet. Run `load_data` method.")

    @data.setter
    def data(self, value: pd.DataFrame):
        """Set the dataset."""
        self._data = LabeledFrame(
            value, y_col=self.target_feature, s_col=self.sensitive_feature
        )

    @property
    def train(self) -> LabeledFrame:
        """Return the training split of the dataset."""
        if self._train is not None:
            return self._train
        else:
            raise ValueError("Data is not loaded yet. Run FolkTables.split_data.")

    @train.setter
    def train(self, value: pd.DataFrame):
        """Set the training split of the dataset."""
        self._train = LabeledFrame(
            value, y_col=self.target_feature, s_col=self.sensitive_feature
        )

    @property
    def validation(self) -> LabeledFrame:
        """Return the validation split of the dataset."""
        if self._validation is not None:
            return self._validation
        else:
            raise ValueError("Data is not loaded yet. Run FolkTables.split_data.")

    @validation.setter
    def validation(self, value: pd.DataFrame):
        """Set the validation split of the dataset."""
        self._validation = LabeledFrame(
            value, y_col=self.target_feature, s_col=self.sensitive_feature
        )

    @property
    def test(self) -> LabeledFrame:
        """Return the test split of the dataset."""
        if self._test is not None:
            return self._test
        else:
            raise ValueError("Data is not loaded yet. Run FolkTables.split_data.")

    @test.setter
    def test(self, value: pd.DataFrame):
        """Set the test split of the dataset."""
        self._test = LabeledFrame(
            value, y_col=self.target_feature, s_col=self.sensitive_feature
        )
