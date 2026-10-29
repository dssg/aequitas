import pickle
import unittest

import numpy as np
import pandas as pd

from aequitas.flow.methods.inprocessing.fairlearn_classifier import FairlearnClassifier


class TestFairlearnClassifier(unittest.TestCase):
    def test_pickle(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x1": rng.normal(size=400), "x2": rng.normal(size=400)})
        s = pd.Series(rng.integers(0, 2, size=400), name="s")
        y = pd.Series(((X["x1"] + 0.5 * s + rng.normal(size=400)) > 0.5).astype(int))

        for reduction, kwargs in [
            ("ExponentiatedGradient", {"max_iter": 3}),
            ("GridSearch", {}),
        ]:
            model = FairlearnClassifier(
                reduction=f"fairlearn.reductions.{reduction}",
                estimator="lightgbm.LGBMClassifier",
                constraint="fairlearn.reductions.FalsePositiveRateParity",
                model__n_estimators=5,
                model__verbose=-1,
                **kwargs,
            )
            model.fit(X, y, s)
            restored = pickle.loads(pickle.dumps(model))
            pd.testing.assert_series_equal(
                restored.predict_proba(X, s), model.predict_proba(X, s)
            )
