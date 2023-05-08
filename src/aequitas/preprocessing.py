import logging
import pandas as pd
from typing import List, Tuple

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


def check_required_cols(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    :param df: A data frame of model results
    :param required_cols: Column names required for selected fairness measures
    :return: None, or ValueError
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def get_attr_cols(df: pd.DataFrame, non_attr_cols: List[str]) -> List[str]:
    """
    :param df: A data frame of model results
    :param non_attr_cols: Names of columns not associated with attributes
    :return: List of columns associated with sample attributes
    """
    attr_cols = df.columns[~df.columns.isin(non_attr_cols)]
    if attr_cols.empty:
        raise ValueError("No attribute columns found in the input data frame.")
    return attr_cols.tolist()


def discretize(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    :param df: A data frame of model results
    :param target_cols: Names of columns to discretize
    :return: A data frame
    """
    for col in target_cols:
        unique_values = df[col].unique()
        if len(unique_values) > 1:
            bins, values = pd.qcut(df[col], 4, precision=2, labels=False, duplicates='drop', retbins=True)
            try:
                df[col] = bins.map(lambda x: '%0.2f' % values[x] + '-' + '%0.2f' % values[x + 1])
            except Exception as e:
                logging.info('Something strange with a column in the input_df: %s', e)
                df = df.drop(col)
        else:
            try:
                df[col] = df[col].astype(str)
            except Exception as e:
                logging.info(f'Something strange with a column in the input_df: {e}')
                df = df.drop(col)
    return df


def preprocess_input_df(df: pd.DataFrame, required_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    :param df: A data frame of model results
    :param required_cols: Names of columns required for bias calculations.
        Default is None.
    :return: A data frame, list of columns associated with sample attributes
    """
    if not required_cols:
        required_cols = ['score']
    try:
        check_required_cols(df, required_cols)
    except ValueError:
        logging.exception('input dataframe does not have all the required columns')
        raise
        logging.error(f'preprocessing.preprocess_input_df: {e}')
        raise

    non_attr_cols = required_cols + ['model_id', 'as_of_date', 'entity_id', 'rank_abs', 'rank_pct', 'id', 'label_value']
    non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str) & (~df.columns.isin(non_attr_cols))]
    df = discretize(df, non_string_cols)

    try:
        attr_cols_input = get_attr_cols(df, non_attr_cols)
    except ValueError as e:
        logging.error(f'preprocessing.preprocess_input_df: {e}')
        raise

    return df, attr_cols_input
