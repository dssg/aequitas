import numpy as np
import pandas as pd
import pytest

from aequitas.fairness import Fairness


@pytest.mark.parametrize("column", ["Supervised Fairness", "Unsupervised Fairness"])
@pytest.mark.parametrize(
    "values, expected",
    [
        ([], "Undefined"),
        ([None, None], "Undefined"),
        ([np.nan, np.nan], "Undefined"),
        ([pd.NA, pd.NA], "Undefined"),
        ([True, None], True),
        ([False, None], False),
        ([True, False, None], False),
        ([True, True], True),
    ],
)
def test_overall_fairness_uses_only_available_determinations(column, values, expected):
    frame = pd.DataFrame({column: values})

    result = Fairness().get_overall_fairness(frame)

    assert result[column] == expected
    assert result["Overall Fairness"] == expected


@pytest.mark.parametrize(
    "supervised, unsupervised, expected",
    [
        ([None], [None], "Undefined"),
        ([True], [None], True),
        ([None], [False], False),
        ([True], [False], False),
    ],
)
def test_overall_fairness_combines_available_categories(
    supervised, unsupervised, expected
):
    frame = pd.DataFrame(
        {"Supervised Fairness": supervised, "Unsupervised Fairness": unsupervised}
    )

    assert Fairness().get_overall_fairness(frame)["Overall Fairness"] == expected


def test_overall_fairness_without_categories_is_undefined():
    assert Fairness().get_overall_fairness(pd.DataFrame()) == {
        "Overall Fairness": "Undefined"
    }
