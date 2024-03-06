import unittest
import os
import pandas as pd
from aequitas.flow.datasets.generic import GenericDataset
from distutils import extension

BASE_DIR = os.path.dirname(__file__)


class TestGenericDataset(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.df = pd.DataFrame(
            {
                "label": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                "sensitive": ["A", "B", "C", "B", "A", "A", "A", "C", "B", "A"],
                "feature1": [0.5, 0.2, 0.7, 0.3, 0.9, 0.5, 0.2, 0.7, 0.3, 0.9],
                "feature2": [0.1, 0.4, 0.6, 0.8, 0.3, 0.5, 0.2, 0.7, 0.3, 0.9],
            }
        )

    def test_load_data_from_dataframe(self):
        dataset = GenericDataset(
            label_column="label", sensitive_column="sensitive", df=self.df
        )
        dataset.load_data()
        self.assertEqual(len(dataset.data), len(self.df))
        self.assertTrue("label" in dataset.data.columns)
        self.assertTrue("sensitive" in dataset.data.columns)

    def test_load_data_from_path_parquet(self):
        dataset = GenericDataset(
            label_column="label",
            sensitive_column="sensitive",
            extension="parquet",
            dataset_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data.parquet"
            ),
        )
        dataset.load_data()
        self.assertEqual(len(dataset.data), 10)
        self.assertTrue("label" in dataset.data.columns)
        self.assertTrue("sensitive" in dataset.data.columns)

    def test_load_data_from_path_csv(self):
        dataset = GenericDataset(
            label_column="label",
            sensitive_column="sensitive",
            extension="csv",
            dataset_path=os.path.join(BASE_DIR, "test_artifacts/test_generic/data.csv"),
        )
        dataset.load_data()
        self.assertEqual(len(dataset.data), 10)
        self.assertTrue("label" in dataset.data.columns)
        self.assertTrue("sensitive" in dataset.data.columns)

    def test_load_data_from_multiple_paths(self):
        dataset = GenericDataset(
            label_column="label",
            sensitive_column="sensitive",
            extension="csv",
            train_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_train.csv"
            ),
            validation_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_validation.csv"
            ),
            test_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_test.csv"
            ),
        )
        dataset.load_data()
        dataset.create_splits()
        self.assertEqual(len(dataset.train), 7)
        self.assertEqual(len(dataset.validation), 2)
        self.assertEqual(len(dataset.test), 1)
        self.assertTrue("label" in dataset.data.columns)
        self.assertTrue("sensitive" in dataset.data.columns)

    def test_create_splits_random(self):
        dataset = GenericDataset(
            label_column="label", sensitive_column="sensitive", df=self.df
        )
        dataset.load_data()
        dataset.create_splits()
        self.assertEqual(len(dataset.train), 7)
        self.assertEqual(len(dataset.validation), 2)
        self.assertEqual(len(dataset.test), 1)

    def test_create_splits_column(self):
        dataset = GenericDataset(
            label_column="label",
            sensitive_column="sensitive",
            df=self.df,
            split_type="column",
            split_column="sensitive",
            split_values={"train": ["A"], "validation": ["B"], "test": ["C"]},
        )
        dataset.create_splits()
        self.assertEqual(len(dataset.train), 5)
        self.assertEqual(len(dataset.validation), 3)
        self.assertEqual(len(dataset.test), 2)


if __name__ == "__main__":
    unittest.main()
