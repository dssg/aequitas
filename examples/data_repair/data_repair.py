import numpy as np
import pandas as pd


def get_group_distributions(
    features: pd.DataFrame, sensitive_attribute: pd.Series, definition: int
) -> tuple[dict, dict]:
    """
    Calculates the quantiles in the dataset for each feature and group.

    Parameters
    ----------
    features : pd.DataFrame
        The feature matrix.
    sensitive_attribute : pd.Series
        The sensitive attribute.

    Returns
    -------
    dict, dict
        The quantiles per group and the global quantiles.
    """
    quantiles = np.linspace(0, 1, definition)
    # Create a dictionary with quantile values per group.
    group_quantiles = {}
    # Also create a dictionary with the global quantiles.
    global_quantiles = {}
    for column in features.columns:
        global_quantiles[column] = features[column].quantile(quantiles).values

        # Get the quantiles for each group.
        group_quantiles[column] = get_group_quantiles(
            features[column], sensitive_attribute, quantiles
        )
    return group_quantiles, global_quantiles


def get_group_quantiles(
    feature: pd.Series,
    sensitive_attribute: pd.Series,
    quantiles: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Transforms the quantiles to a more digested value.

    The original method of pandas creates a dictionary with a two-level
    index, which is harder to query. This transforms the two level index in
    dictionaries within a single dictionary.

    Parameters
    ----------
    feature : pd.Series
        The feature to obtain the quantiles.
    sensitive_attribute : pd.Series
        The sensitive attribute.
    quantiles : np.ndarray
        The quantiles to calculate.
    """
    quantile_dict = feature.groupby(sensitive_attribute).quantile(quantiles)
    transformed_quantile_dict = {}
    for group in quantile_dict.index.get_level_values(0).unique():
        transformed_quantile_dict[group] = quantile_dict[group].values
    return transformed_quantile_dict


def repair_features(
    features: pd.DataFrame,
    sensitive_attribute: pd.Series,
    definition: int,
    repair_level: float,
    group_quantiles: dict,
    global_quantiles: dict,
) -> pd.DataFrame:
    """
    Transform the features conditioned of protected attribute to match the global
    distribution.

    Parameters
    ----------
    features : pd.DataFrame
        The features.
    sensitive_attribute : pd.Series
        The sensitive attribute.
    definition : int
        The number of quantiles to use.
    repair_level : float
        The level of repair to apply. 0 means no repair, 1 means full repair.
    group_quantiles : dict
        The quantiles per group.
    global_quantiles : dict
        The global quantiles.
    """
    features_repaired = features.copy()

    quantile_points = np.linspace(0, 1, definition)
    for column in global_quantiles.keys():
        # Calculate the quantile of every point for every group (vectorized)
        interpolation_quantiles = {}
        # Also calculate the global distribution value at that quantile
        global_values = {}
        # After, calculate the corrected value for every point
        corrected_values = {}
        for group in sensitive_attribute.unique():
            interpolation_quantiles[group] = (
                np.interp(
                    features_repaired[column],
                    group_quantiles[column][group],
                    quantile_points,
                )
                + np.interp(
                    -features_repaired[column],
                    -group_quantiles[column][group][::-1],
                    quantile_points[::-1],
                )
            ) / 2
            global_values[group] = np.interp(
                interpolation_quantiles[group],
                quantile_points,
                global_quantiles[column],
            )
            corrected_values[group] = global_values[
                group
            ] * repair_level + features_repaired[column].values * (1 - repair_level)

        repaired_column = [
            corrected_values[group][index]
            for index, group in enumerate(sensitive_attribute)
        ]
        features_repaired[column] = repaired_column
    return features_repaired
