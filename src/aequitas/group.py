import logging
import warnings
import pandas as pd
import numpy as np
import uuid
logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Benedict Kuester, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


COLUMN_ORDER = ['model_id', 'score_threshold', 'k', 'attribute_name',
                'attribute_value', 'tpr', 'tnr', 'for', 'fdr', 'fpr', 'fnr',
                'npv', 'precision', 'pp', 'pn', 'ppr', 'pprev', 'fp', 'fn',
                'tn', 'tp', 'group_label_pos', 'group_label_neg', 'group_size',
                'total_entities', 'prev']


def gen_metrics_df(df, attr_cols, score, score_threshold):
    divide = lambda x, y: x / y if y != 0 else np.nan
    model_id = 0
    columns = [
        'model_id', 'score_threshold', 'k', 'attribute_name', 'attribute_value',
       'tpr', 'tnr', 'fomr', 'fdr', 'fpr', 'fnr', 'npv', 'precision', 'pp',
       'pn', 'ppr', 'pprev', 'fp', 'fn', 'tn', 'tp', 'group_label_pos',
       'group_label_neg', 'group_size', 'total_entities', 'prev']
    final_dict = {column: [] for column in columns}
    k = df[df[score]==1].shape[0]
    for attribute_name in attr_cols:
        grouped_data = df.groupby(by=[attribute_name, score, "label_value",]).size().to_dict()
        attribute_values = set([key[0] for key in grouped_data.keys()])
        for attribute_value in attribute_values:
            tp = grouped_data.get((attribute_value, 1, 1), 0)
            tn = grouped_data.get((attribute_value, 0, 0), 0)
            fp = grouped_data.get((attribute_value, 1, 0), 0)
            fn = grouped_data.get((attribute_value, 0, 1), 0)
            tpr = divide(tp, fn + tp)
            tnr = divide(tn, fp + tn)
            fomr = divide(fn, fn + tn)
            fdr = divide(fp, tp + fp)
            fpr = divide(fp, fp + tn)
            fnr = divide(fn, fn + tp)
            npv = divide(tn, fn + tn)
            precision = divide(tp, tp + fp)
            pp = fp + tp
            pn = fn + tn
            ppr = divide(pp, k)
            pprev = divide (pp, pp + pn)
            group_label_pos = (fn + tp)
            group_label_neg = (fp + tn)
            group_size = group_label_pos + group_label_neg
            total_entities = df.shape[0]
            prev = (fn + tp) / group_size
            for key in final_dict.keys():
                final_dict[key].append(locals()[key])
    return pd.DataFrame(final_dict).rename(columns={"fomr": "for"})


class Group(object):
    """
    """
    all_group_metrics = ('ppr', 'pprev', 'precision', 'fdr', 'for', 'fpr',
                         'fnr', 'tpr', 'tnr', 'npv', 'prev')

    def __init__(self, input_group_metrics=all_group_metrics):
        """
        """
        self.absolute_metrics = input_group_metrics
        self.label_neg_count = lambda label_col: lambda x: \
            (x[label_col] == 0).sum()
        self.label_pos_count = lambda label_col: lambda x: \
            (x[label_col] == 1).sum()
        self.confusion_matrix_functions = self.get_confusion_matrix_functions()

    @staticmethod
    def get_confusion_matrix_functions():
        false_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 1)).sum()

        false_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 0)).sum()

        true_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 0)).sum()

        true_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 1)).sum()
        return {
            'fp': false_pos_count,
            'fn': false_neg_count,
            'tn': true_neg_count,
            'tp': true_pos_count
        }

    def _check_model_id(self, df, method_table_name):
        if 'model_id' in df.columns:
            df_models = df.model_id.unique()
            if len(df_models) != 1:
                raise ValueError('This method requires one and only one model_id in the dataframe. '
                                 f'Tip: Check that {method_table_name}.model_id.unique() returns a one-element array. ')
            else:
                return df_models[0]
        else:
            return 0

    def get_multimodel_crosstabs(self, df, score_thresholds=None, attr_cols=None):
        """
        Calls `get_crosstabs()` for univariate groups and calculates group
        metrics for results from multiple models.

        :param df: a dataframe containing the following required columns [score,  label_value].
        :param score_thresholds: dictionary { 'rank_abs':[] , 'rank_pct':[], 'score':[] }
        :param attr_cols: optional, list of names of columns corresponding to
            group attributes (i.e., gender, age category, race, etc.).

        :return: A dataframe of group score, label, and error statistics and absolute bias metric values grouped by unique attribute values
        """
        if 'model_id' not in df.columns:
            raise ValueError("This method expects at least two distinct 'model_id' values "
                             f"in the dataframe. Tip: Check that 'df'' has a column called 'model_id.'")

        df_models = df.model_id.unique()
        crosstab_list = []

        if len(df_models) > 1:
            for model in df_models:
                model_df = df.loc[df['model_id'] == model]
                model_crosstab, model_attr_cols = self.get_crosstabs(model_df, score_thresholds=score_thresholds, attr_cols=attr_cols)
                crosstab_list.append(model_crosstab)

            # Note: only returns model_attr_cols from last iteration, as all will be same
            return pd.concat(crosstab_list, ignore_index=True), model_attr_cols
        else:
            return self.get_crosstabs(df, score_thresholds=score_thresholds, attr_cols=attr_cols)

    def get_crosstabs(self, df, score_thresholds=None, attr_cols=None):
        """
        Creates univariate groups and calculates group metrics for results
        from a single model.

        :param df: a dataframe containing the following required columns [score,  label_value].
        :param score_thresholds: dictionary { 'rank_abs':[] , 'rank_pct':[], 'score_val': []}
        :param attr_cols: optional, list of names of columns corresponding to
            group attributes (i.e., gender, age category, race, etc.).

        :return: A dataframe of group score, label, and error statistics and absolute bias metric values grouped by unique attribute values
        """
        df = df.copy()  # To not transform original dataframe.
        if not attr_cols:
            non_attr_cols = ['id', 'model_id', 'entity_id', 'score',
                             'label_value']
            attr_cols = df.columns[~df.columns.isin(
                non_attr_cols)]  # index of the columns that are

        # Validation step.
        for col in attr_cols:
            # Validate if column is in dataframe.
            if col not in df.columns:
                raise KeyError(
                    f'The attribute column \'{col}\' is not on the dataframe.')
            # Validate if column has correct datatype.
            if df[col].dtype not in (object, str):
                raise TypeError(
                    f'The attribute column \'{col}\' has in invalid datatype.')
        if len(df['label_value'].unique()) > 2:
            raise ValueError('Labels are not binarized.')
        else:
            df['label_value'] = df['label_value'].astype(int)

        count_ones = None  # it also serves as flag to set parameter to 'binary'
        confusion_matrixes = []
        binarized_scores = False
        if not score_thresholds and len(df['score'].unique()) > 2:
            raise ValueError(f'Scores are not binarized. Provide a threshold.')
        elif not score_thresholds:  # No thresholds given and binarized
            df['score'] = df['score'].astype(int)
            confusion_matrixes.append(
                gen_metrics_df(df, attr_cols, 'score', 'binary 0/1'))
            score_thresholds = {}

        binarized_column = uuid.uuid4()  # Add a column with the binarized scores
        sorted_df = False  # Flag to sort the dataframe in the first run
        for key, values in score_thresholds.items():
            for value in values:
                if not sorted_df:
                    df = df.sort_values('score', ascending=False).reset_index(
                        drop=True)
                    sorted_df = True
                if key == 'rank_abs':
                    df[binarized_column] = (df.index < value).astype(int)
                elif key == 'rank_pct':
                    df[binarized_column] = (
                                df.index < value * df.shape[0]).astype(int)
                elif key == 'score_val':
                    df[binarized_column] = (df['score_val'] >= value).astype(
                        int)
                else:
                    raise KeyError(f'Invalid keys')
                confusion_matrixes.append(
                    gen_metrics_df(df, attr_cols, binarized_column,
                                   f'{value}_{key[-3:]}'))
        return pd.concat(confusion_matrixes), attr_cols

    def list_absolute_metrics(self, df):
        """
        View list of all calculated absolute bias metrics in df
        """
        return df.columns.intersection(self.absolute_metrics).tolist()
