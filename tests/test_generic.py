import unittest
import os
import pandas as pd
from aequitas.flow.datasets.generic import GenericDataset

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

    def test_create_splits_column_from_path(self):
        dataset = GenericDataset(
            label_column="label",
            sensitive_column="sensitive",
            dataset_path=os.path.join(BASE_DIR, "test_artifacts/test_generic/data.csv"),
            split_type="column",
            extension="csv",
            split_column="sensitive",
            split_values={"train": ["A"], "validation": ["B"], "test": ["C"]},
        )
        dataset.load_data()
        dataset.create_splits()
        self.assertEqual(len(dataset.train), 5)
        self.assertEqual(len(dataset.validation), 3)
        self.assertEqual(len(dataset.test), 2)

    def test_all_paths_provided(self):
        self.assertRaisesRegex(
            ValueError,
             "If single dataset path is passed, the other paths must be None.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            train_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_train.csv"
            ),
            validation_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_validation.csv"
            ),
            test_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_test.csv"
            ),
            dataset_path=os.path.join(BASE_DIR, "test_artifacts/test_generic/data.csv"),
        )

    def test_missing_paths(self):
        self.assertRaisesRegex(
            ValueError,
            "If multiple dataset paths are passed, the single path must be" "`None`.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            train_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_train.csv"
            ),
            validation_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_validation.csv"
            ),
        )

    def test_invalid_path(self):
        self.assertRaisesRegex(
            ValueError,
            "Invalid path:*",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            train_path=os.path.join(BASE_DIR, "test_artifacts/data_train.csv"),
            validation_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_validation.csv"
            ),
            test_path=os.path.join(
                BASE_DIR, "test_artifacts/test_generic/data_test.csv"
            ),
        )

    def test_missing_split_key(self):
        self.assertRaisesRegex(
            ValueError,
            "Missing key in passed splits: test",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            dataset_path=os.path.join(BASE_DIR, "test_artifacts/test_generic/data.csv"),
            split_values={"train": 0.63, "validation": 0.37},
        )

    def test_invalid_splits(self):
        self.assertRaisesRegex(
            ValueError,
            "Invalid split sizes. Make sure the sum of proportions for all the"
            " datasets is equal to or lower than 1.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            dataset_path=os.path.join(BASE_DIR, "test_artifacts/test_generic/data.csv"),
            split_values={"train": 0.63, "validation": 0.37, "test": 0.2},
        )

    def test_invalid_splits_warn(self):
        with self.assertLogs("datasets.GenericDataset", level="WARN") as cm:
            dataset = GenericDataset(
                label_column="label",
                sensitive_column="sensitive",
                dataset_path=os.path.join(
                    BASE_DIR, "test_artifacts/test_generic/data.csv"
                ),
                split_values={"train": 0.3, "validation": 0.1, "test": 0.2},
            )
        self.assertEqual(
            cm.output,
            [
                "WARNING:datasets.GenericDataset:Using only 0.6000000000000001 of the dataset."
            ],
        )

    def test_missing_splits_column(self):
        self.assertRaisesRegex(
            ValueError,
            "Split column must be specified when using column split.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            df=self.df,
            split_type="column",
            split_values={"train": ["A"], "validation": ["B"], "test": ["C"]},
        )

    def test_wrong_splits_column(self):
        self.assertRaisesRegex(
            ValueError,
            "Split column must be a column in the dataset.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            df=self.df,
            split_type="column",
            split_column="test",
            split_values={"train": ["A"], "validation": ["B"], "test": ["C"]},
        )

    def test_wrong_splits_value(self):
        self.assertRaisesRegex(
            ValueError,
            "Split values must be present in the split column.",
            GenericDataset,
            label_column="label",
            sensitive_column="sensitive",
            split_type="column",
            df=self.df,
            split_column="sensitive",
            split_values={"train": ["D"], "validation": ["B"], "test": ["C"]},
        )


if __name__ == "__main__":
    unittest.main()
