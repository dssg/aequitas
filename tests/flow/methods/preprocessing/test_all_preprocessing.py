import unittest

from aequitas.flow.datasets import FolkTables

from aequitas.flow.methods.preprocessing import (
    CorrelationSuppression,
    DataRepairer,
    FeatureImportanceSuppression,
    Identity,
    LabelFlipping,
    Massaging,
    PrevalenceSampling,
)

METHODS = [
    CorrelationSuppression,
    DataRepairer,
    FeatureImportanceSuppression,
    Identity,
    LabelFlipping,
    Massaging,
    PrevalenceSampling,
]


class TestAllPreprocessing(unittest.TestCase):
    methods = []

    def setUp(self):
        self.dataset = FolkTables("ACS_sample")
        self.dataset.load_data()
        self.dataset.create_splits()

    def test_0_instantiate_methods(self):
        for method in METHODS:
            method_instance = method()
            self.assertTrue(hasattr(method_instance, "used_in_inference"))
            self.assertTrue(hasattr(method_instance, "fit"))
            self.assertTrue(hasattr(method_instance, "transform"))
            self.methods.append(method_instance)

    def test_1_fit_methods(self):
        for method in self.methods:
            method.fit(self.dataset.train.X, self.dataset.train.y, self.dataset.train.s)

    def test_2_transform_methods(self):
        for method in self.methods:
            method.transform(
                self.dataset.train.X, self.dataset.train.y, self.dataset.train.s
            )
