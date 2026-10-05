"""Documented sensitive-attribute types must produce equivalent bias audits."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aequitas import Audit
from aequitas.group import Group


def _frame():
    return pd.DataFrame(
        {
            "score": [0, 1, 1, 1, 1, 0, 0, 0],
            "label_value": [0, 1, 0, 1, 1, 0, 0, 1],
            "group": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )


@pytest.mark.parametrize("selection", [None, "group", ["group"]])
@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_audit_attribute_types_and_selection_match_object_baseline(selection, dtype):
    expected = Audit(_frame(), sensitive_attribute_column=["group"])
    expected.audit()
    frame = _frame()
    frame["group"] = frame["group"].astype(dtype)
    original = frame.copy(deep=True)

    audit = Audit(frame, sensitive_attribute_column=selection)
    assert audit.sensitive_attribute_column == ["group"]
    audit.audit()

    assert_frame_equal(audit.metrics_df, expected.metrics_df)
    assert_frame_equal(audit.disparity_df, expected.disparity_df)
    metrics = audit.metrics_df.set_index("attribute_value")
    assert metrics.loc["a", "pp"] == 3
    assert metrics.loc["b", "pp"] == 1
    assert metrics.loc["a", "tpr"] == 1.0
    assert metrics.loc["b", "tpr"] == 0.5
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("numeric_categories", [False, True])
def test_categorical_audit_ignores_unused_categories(ordered, numeric_categories):
    frame = _frame()
    if numeric_categories:
        frame["group"] = frame["group"].map({"a": 1, "b": 2})
        categories = [2, 1, 3]
    else:
        categories = ["b", "a", "unused"]
    expected = Audit(
        frame.assign(group=frame["group"].astype(object)),
        sensitive_attribute_column=["group"],
    )
    expected.audit()
    frame["group"] = pd.Categorical(
        frame["group"], categories=categories, ordered=ordered
    )
    original = frame.copy(deep=True)

    audit = Audit(frame, sensitive_attribute_column=["group"])
    audit.audit()

    assert_frame_equal(audit.metrics_df, expected.metrics_df)
    assert_frame_equal(audit.disparity_df, expected.disparity_df)
    assert len(audit.metrics_df) == 2
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("selection", [None, ["group", "second_group"]])
def test_multiple_sensitive_attributes_keep_inferred_columns(selection):
    frame = _frame()
    frame["second_group"] = ["x", "y"] * 4
    expected = Audit(
        frame, sensitive_attribute_column=["group", "second_group"]
    )
    expected.audit()
    frame["group"] = frame["group"].astype("category")
    frame["second_group"] = frame["second_group"].astype("string")

    audit = Audit(frame, sensitive_attribute_column=selection)
    audit.audit()

    assert audit.sensitive_attribute_column == ["group", "second_group"]
    assert_frame_equal(audit.metrics_df, expected.metrics_df)
    assert_frame_equal(audit.disparity_df, expected.disparity_df)


@pytest.mark.parametrize("dtype", ["category", "string"])
@pytest.mark.parametrize("thresholds", [None, {"score_val": [0.5]}])
def test_group_crosstabs_support_extension_attributes_without_mutating_input(
    dtype, thresholds
):
    frame = _frame()
    if thresholds is not None:
        frame["score"] = [0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
    expected, _ = Group().get_crosstabs(
        frame, attr_cols=["group"], score_thresholds=thresholds
    )
    frame["group"] = frame["group"].astype(dtype)
    original = frame.copy(deep=True)

    actual, columns = Group().get_crosstabs(
        frame, attr_cols=["group"], score_thresholds=thresholds
    )

    assert columns == ["group"]
    assert_frame_equal(actual, expected)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["int64", "float64", "bool"])
def test_noncategorical_numeric_attributes_are_still_rejected(dtype):
    frame = _frame()
    frame["group"] = pd.Series([0, 1] * 4, dtype=dtype)

    with pytest.raises(ValueError, match="must be categorical"):
        Audit(frame, sensitive_attribute_column=["group"])
    with pytest.raises(TypeError, match="invalid datatype"):
        Group().get_crosstabs(frame, attr_cols=["group"])


@pytest.mark.parametrize("selection", ["missing", ["group", "missing"]])
def test_missing_sensitive_attribute_names_are_still_rejected(selection):
    with pytest.raises(ValueError, match="not found in the DataFrame"):
        Audit(_frame(), sensitive_attribute_column=selection)


def test_no_sensitive_attributes_is_still_rejected():
    with pytest.raises(ValueError, match=r"Sensitive attribute column\(s\) not found"):
        Audit(_frame().drop(columns="group"))
