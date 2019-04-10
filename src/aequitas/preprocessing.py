import logging
from sys import exit

import pandas as pd

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

def check_required_cols(df, required_cols):
    """

    :param df: A data frame of model results
    :param required_cols: Column names required for selected fairness measures
    :return: None, or ValueError
    """
    check_model_cols = [col in df.columns for col in required_cols]
    if False in check_model_cols:
        raise ValueError
    return


def get_attr_cols(df, non_attr_cols):
    """
    :param df: A data frame of model results
    :param non_attr_cols: Names of columns not associated with attributes
    :return: List of columns associated with sample attributes
    """
    # index of the columns that are associated with attributes
    attr_cols = df.columns[~df.columns.isin(non_attr_cols)]
    if attr_cols.empty:
        raise ValueError
    return attr_cols.tolist()


def discretize(df, target_cols):
    """

    :param df: A data frame of model results
    :param target_cols: Names of columns to discretize
    :return: A data frame
    """
    for col in target_cols:
        if len(df[col].unique()) > 1:
            bins, values = pd.qcut(df[col], 4, precision=2, labels=False, duplicates='drop', retbins=True)
            try:
                df[col] = bins.map(lambda x: '%0.2f' % values[x] + '-' + '%0.2f' % values[x + 1])
            except Exception as e:
                logging.info('Something strange with a column in the input_df ' + str(e))
                df = df.drop(col)
        else:
            try:
                df[col] = df[col].astype(str)
            except Exception as e:
                logging.info('Something strange with a column in the input_df ' + str(e))
                df = df.drop(col)
    return df


def preprocess_input_df(df, required_cols=None):
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
        logging.error('preprocessing.preprocess_input_df: input dataframe does not have all the required columns.')
    non_attr_cols = required_cols + ['model_id', 'as_of_date', 'entity_id', 'rank_abs', 'rank_pct', 'id', 'label_value']
    non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str) & (~df.columns.isin(non_attr_cols))]
    df = discretize(df, non_string_cols)
    try:
        attr_cols_input = get_attr_cols(df, non_attr_cols)
    except ValueError:
        logging.error('preprocessing.preprocess_input_df: input dataframe does not have any other columns besides required '
                      'columns. Please add attribute columns to the input df.')
    return df, attr_cols_input
