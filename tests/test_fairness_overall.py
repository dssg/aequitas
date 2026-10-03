"""Unit tests for Fairness.get_overall_fairness edge cases."""
import pytest
import pandas as pd
import numpy as np
from aequitas.fairness import Fairness


def test_overall_fairness_returns_undefined_for_all_nan():
    """Case A: all NaN Unsupervised Fairness -> 'Undefined' (was returning True)."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'],
        'score_threshold': [0.5],
        'attribute_name': ['race'],
        'Unsupervised Fairness': [np.nan],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] == 'Undefined'
    assert result['Overall Fairness'] == 'Undefined'


def test_overall_fairness_returns_undefined_for_multiple_nan():
    """Case A2: multiple NaN rows, no actual parity values -> 'Undefined'."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'] * 3,
        'score_threshold': [0.5] * 3,
        'attribute_name': ['race'] * 3,
        'Unsupervised Fairness': [np.nan, np.nan, np.nan],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] == 'Undefined'
    assert result['Overall Fairness'] == 'Undefined'


def test_overall_fairness_returns_true_when_all_true():
    """Case C: all groups in band -> True."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'] * 2,
        'score_threshold': [0.5] * 2,
        'attribute_name': ['race'] * 2,
        'Unsupervised Fairness': [True, True],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] is True
    assert result['Overall Fairness'] is True


def test_overall_fairness_returns_false_when_any_false():
    """Case B: one group outside band -> False."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'] * 2,
        'score_threshold': [0.5] * 2,
        'attribute_name': ['race'] * 2,
        'Unsupervised Fairness': [True, False],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] is False
    assert result['Overall Fairness'] is False


def test_overall_fairness_supervised_undefined():
    """Case G: Supervised True but Unsupervised all-NaN -> Overall=True."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'],
        'score_threshold': [0.5],
        'attribute_name': ['race'],
        'Unsupervised Fairness': [np.nan],
        'Supervised Fairness': [True],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] == 'Undefined'
    assert result['Supervised Fairness'] is True
    assert result['Overall Fairness'] is True


def test_overall_fairness_mixed_supervised():
    """Unsupervised Undefined, Supervised False -> Overall=False."""
    f = Fairness()
    df = pd.DataFrame({
        'model_id': ['model_a'] * 2,
        'score_threshold': [0.5] * 2,
        'attribute_name': ['race'] * 2,
        'Unsupervised Fairness': [np.nan, np.nan],
        'Supervised Fairness': [True, False],
    })
    result = f.get_overall_fairness(df)
    assert result['Unsupervised Fairness'] == 'Undefined'
    assert result['Supervised Fairness'] is False
    assert result['Overall Fairness'] is False
