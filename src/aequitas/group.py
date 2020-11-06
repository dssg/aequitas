import logging
import warnings
import pandas as pd
import numpy as np
logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Benedict Kuester, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


COLUMN_ORDER = ['model_id', 'score_threshold', 'k', 'attribute_name',
                'attribute_value', 'tpr', 'tnr', 'for', 'fdr', 'fpr', 'fnr',
                'npv', 'precision', 'pp', 'pn', 'ppr', 'pprev', 'fp', 'fn',
                'tn', 'tp', 'group_label_pos', 'group_label_neg', 'group_size',
                'total_entities', 'prev']

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
        self.group_functions = self._get_group_functions()
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

    @staticmethod
    def _get_group_functions():
        """
        Helper function to accumulate lambda functions used in bias metrics
        calculations.
        """

        divide = lambda x, y: x / y if y != 0 else np.nan

        predicted_pos_count = lambda k: lambda x: x['fp'] + x['tp']

        predicted_neg_count = lambda k: lambda x: x['fn'] + x['tn']

        predicted_pos_ratio_k = lambda k: lambda x: divide(x['fp'] + x['tp'], k)

        predicted_pos_ratio_g = lambda k: lambda x: divide(
            x['fp'] + x['tp'], x['fn'] + x['tn'] + x['fp'] + x['tp']
        )

        fpr = lambda k: lambda x: divide(x['fp'], x['fp'] + x['tn'])

        tnr = lambda k: lambda x: divide(x['tn'], x['fp'] + x['tn'])

        fnr = lambda k: lambda x: divide(x['fn'], x['fn'] + x['tp'])

        tpr = lambda k: lambda x: divide(x['tp'], x['fn'] + x['tp'])

        fomr = lambda k: lambda x: divide(x['fn'], x['fn'] + x['tn'])

        npv = lambda k: lambda x: divide(x['tn'], x['fn'] + x['tn'])

        precision = lambda k: lambda x: divide(x['tp'], x['tp'] + x['fp'])

        fdr = lambda k: lambda x: divide(x['fp'], x['tp'] + x['fp'])

        group_functions = {'tpr': tpr,
                           'tnr': tnr,
                           'for': fomr,
                           'fdr': fdr,
                           'fpr': fpr,
                           'fnr': fnr,
                           'npv': npv,
                           'precision': precision,
                           'pp': predicted_pos_count,
                           'pn': predicted_neg_count,
                           'ppr': predicted_pos_ratio_k,
                           'pprev': predicted_pos_ratio_g,
                           }

        return group_functions



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
        :param score_thresholds: dictionary { 'rank_abs':[] , 'rank_pct':[], 'score':[] }
        :param attr_cols: optional, list of names of columns corresponding to
            group attributes (i.e., gender, age category, race, etc.).

        :return: A dataframe of group score, label, and error statistics and absolute bias metric values grouped by unique attribute values
        """
        model_id = self._check_model_id(df, method_table_name='df')

        if not attr_cols:
            non_attr_cols = ['id', 'model_id', 'entity_id', 'score', 'label_value', 'rank_abs', 'rank_pct']
            attr_cols = df.columns[~df.columns.isin(non_attr_cols)]  # index of the columns that are

        df_cols = set(df.columns)
        # check if all attr_cols exist in df
        if len(set(attr_cols) - df_cols) > 0:
            raise Exception('get_crosstabs: not all attribute columns provided exist in input dataframe!')

        # check if all columns are strings:
        non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str) & (df.columns.isin(attr_cols))]
        if non_string_cols.empty is False:
            raise Exception('get_crosstabs: input df was not preprocessed. There are non-string cols within attr_cols!')

        # if no score_thresholds are provided, we assume that rank_abs=number of 1s in the score column
        count_ones = None  # it also serves as flag to set parameter to 'binary'

        if not score_thresholds:
            df.loc[:, 'score'] = df.loc[:,'score'].astype(float)
            count_ones = df['score'].value_counts().get(1.0, 0)
            score_thresholds = {'rank_abs': [count_ones]}

        df = df.sort_values('score', ascending=False)
        df['rank_abs'] = range(1, len(df) + 1)
        df['rank_pct'] = df['rank_abs'] / len(df)
        dfs = []
        prior_dfs = []
        # calculate the bias for these columns
        # not default(non_attr_cols), therefore represent the group variables!
        logging.info('getcrosstabs: attribute columns to perform crosstabs:' + ','.join(attr_cols))
        # for each group variable do
        for col in attr_cols:
            # find the priors_df
            col_group = df.fillna({col: np.nan}).groupby(col)
            counts = col_group.size()
            # distinct entities within group value
            this_prior_df = pd.DataFrame({
                'model_id': [model_id] * len(counts),
                'attribute_name': [col] * len(counts),
                'attribute_value': counts.index.values,
                'group_label_pos': col_group.apply(self.label_pos_count(
                    'label_value')).values,
                'group_label_neg': col_group.apply(self.label_neg_count(
                    'label_value')).values,
                'group_size': counts.values,
                'total_entities': [len(df)] * len(counts)
            })
            this_prior_df['prev'] = this_prior_df['group_label_pos'] / this_prior_df['group_size']
            # for each model_id and as_of_date the priors_df has length
            # attribute_names * attribute_values
            prior_dfs.append(this_prior_df)

            # we calculate the bias for two different types of score_thresholds
            # units (percentage ranks and absolute ranks)
            # YAML ex: thresholds:
            #              rank_abs: [300]
            #              rank_pct: [0.01, 0.02, 0.05, 0.10]
            for thres_unit, thres_values in score_thresholds.items():

                for thres_val in thres_values:
                    flag = 0
                    k = (df[thres_unit] <= thres_val).sum()

                    # denote threshold as binary if numeric count_ones value
                    # donate as [rank value]_abs or [rank_value]_pct otherwise
                    score_threshold = 'binary 0/1' if count_ones != None else str(thres_val) + '_' + thres_unit[-3:]
                    for name, func in self.confusion_matrix_functions.items():
                        func = func(thres_unit, 'label_value', thres_val, k)
                        feat_bias = col_group.apply(func)
                        metrics_df = pd.DataFrame({
                            'model_id': [model_id] * len(feat_bias),
                            'score_threshold': [score_threshold] * len(feat_bias),
                            'k': [k] * len(feat_bias),
                            'attribute_name': [col] * len(feat_bias),
                            'attribute_value': feat_bias.index.values,
                            name: feat_bias.values
                        })
                        if flag == 0:
                            this_group_df = metrics_df
                            flag = 1
                        else:
                            this_group_df = this_group_df.merge(metrics_df)
                    for name, func in self.group_functions.items():
                        func = func(k)
                        feat_bias = this_group_df.apply(func, axis=1)
                        this_group_df[name] = feat_bias
                    dfs.append(this_group_df)
        groups_df = pd.concat(dfs, ignore_index=True)
        priors_df = pd.concat(prior_dfs, ignore_index=True)
        groups_df = groups_df.merge(priors_df, on=['model_id', 'attribute_name',
                                                   'attribute_value'])
        return groups_df[COLUMN_ORDER], attr_cols


    def list_absolute_metrics(self, df):
        """
        View list of all calculated absolute bias metrics in df
        """
        return df.columns.intersection(self.absolute_metrics).tolist()
