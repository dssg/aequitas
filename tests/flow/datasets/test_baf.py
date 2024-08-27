import unittest
from aequitas.flow.datasets.baf import BankAccountFraud, VARIANTS, DEFAULT_PATH


# TODO: These tests can be merged with the ones in test_folktables.py

class TestBankAccountFraudDataset(unittest.TestCase):
    # Test loading related functionalities.
    def test_load_variants(self):
        for variant in VARIANTS:
            dataset = BankAccountFraud(variant)
            dataset.load_data()
            self.assertTrue(len(dataset.data) > 0)
            self.assertTrue("customer_age_bin" in dataset.data.columns)
            self.assertTrue("fraud_bool" in dataset.data.columns)

    def test_load_invalid_variant(self):
        with self.assertRaises(ValueError):
            BankAccountFraud("invalid_variant")

    def test_download(self):
        # Remove default folder of datasets even if not empty
        if DEFAULT_PATH.exists():
            for file in DEFAULT_PATH.iterdir():
                file.unlink()
            DEFAULT_PATH.rmdir()
        for variant in VARIANTS:
            dataset = BankAccountFraud(variant)
            dataset.load_data()
            self.assertTrue(dataset.path.exists())

    # Test split related functionalities.
    def test_invalid_split_type(self):
        with self.assertRaises(ValueError):
            BankAccountFraud(VARIANTS[0], split_type="invalid_split_type")

    def test_default_split(self):
        dataset = BankAccountFraud(VARIANTS[0])
        dataset.load_data()
        dataset.create_splits()
        self.assertTrue(len(dataset.train) > 0)
        self.assertTrue(len(dataset.test) > 0)
        self.assertTrue(len(dataset.validation) > 0)

    def test_random_split(self):
        dataset = BankAccountFraud(
            VARIANTS[0],
            split_type="random",
            splits={"train": 0.6, "validation": 0.2, "test": 0.2},
        )
        dataset.load_data()
        dataset.create_splits()
        self.assertTrue(len(dataset.train) > 0)
        self.assertTrue(len(dataset.test) > 0)
        self.assertTrue(len(dataset.validation) > 0)

    def test_invalid_random_split_missing_key(self):
        with self.assertRaises(ValueError):
            BankAccountFraud(
                VARIANTS[0],
                split_type="random",
                splits={"train": 0.6, "validation": 0.2},
            )

    def test_invalid_random_split_more_than_1(self):
        with self.assertRaises(ValueError):
            BankAccountFraud(
                VARIANTS[0],
                split_type="random",
                splits={"train": 0.6, "validation": 0.2, "test": 0.3},
            )

    # Test sensitive column related issues.
    def test_housing_sensitive_column(self):
        dataset = BankAccountFraud(VARIANTS[0], sensitive_column="housing_status")
        dataset.load_data()
        self.assertTrue("housing_status" in dataset.data.columns)
        self.assertTrue(dataset.data.s.name == "housing_status")

    def test_invalid_sensitive_column(self):
        with self.assertRaises(ValueError):
            BankAccountFraud(VARIANTS[0], sensitive_column="invalid_column")

    def test_invalid_sensitive_column_type(self):
        with self.assertRaises(ValueError):
            BankAccountFraud(VARIANTS[0], sensitive_column="name_email_similarity")
            # Numerical column


if __name__ == "__main__":
    unittest.main()
