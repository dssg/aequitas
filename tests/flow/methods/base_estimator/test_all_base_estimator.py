import unittest

from aequitas.flow.datasets import FolkTables

from aequitas.flow.methods.base_estimator import (
    LightGBM,
    LogisticRegression,
    RandomForest,
)

METHODS = [
    LightGBM,
    LogisticRegression,
    RandomForest,
]

# TODO: Add NeuralNetwork to the list of methods

class TestAllBaseEstimator(unittest.TestCase):
    methods = []

    def setUp(self):
        self.dataset = FolkTables("ACS_sample")
        self.dataset.load_data()
        self.dataset.create_splits()

    def test_0_instantiate_methods(self):
        for method in METHODS:
            method_instance = method()
            self.assertTrue(hasattr(method_instance, "fit"))
            self.assertTrue(hasattr(method_instance, "predict_proba"))
            self.assertTrue(hasattr(method_instance, "_unawareness"))
            self.methods.append(method_instance)

    def test_1_fit_methods(self):
        for method in self.methods:
            method.fit(self.dataset.train.X, self.dataset.train.y, self.dataset.train.s)

    def test_2_transform_methods(self):
        for method in self.methods:
            method.predict_proba(self.dataset.validation.X, self.dataset.validation.s)
