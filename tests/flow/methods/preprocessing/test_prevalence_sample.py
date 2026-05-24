import unittest

from aequitas.flow.datasets import FolkTables

from aequitas.flow.methods.preprocessing import PrevalenceSampling


class TestPrevalenceSampling(unittest.TestCase):
    def setUp(self):
        self.dataset = FolkTables("ACS_sample")
        self.dataset.load_data()
        self.dataset.create_splits()

    def test_invalid_strategy_error(self):
        with self.assertRaises(ValueError):
            PrevalenceSampling(strategy="invalid_strategy")

    def test_invalid_strategy_error_in_internal_method(self):
        method = PrevalenceSampling()
        with self.assertRaises(ValueError):
            method._calculate_sample_sizes(1, 1, 1, "invalid_strategy", 0.05)

    def test_reference_as_most_prevalent(self):
        method = PrevalenceSampling(s_ref=None)
        method.fit(self.dataset.train.X, self.dataset.train.y, self.dataset.train.s)
        self.assertEqual(method.s_ref, self.dataset.train.s.mode().values[0])

    def test_oversampling(self):
        method = PrevalenceSampling(strategy="oversample")
        method.fit(self.dataset.train.X, self.dataset.train.y, self.dataset.train.s)
        method.transform(
            self.dataset.train.X,
            self.dataset.train.y,
            self.dataset.train.s,
        )
