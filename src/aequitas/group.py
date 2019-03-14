import logging

import pandas as pd

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Benedict Kuester, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


class Group(object):
    """
    """
    def __init__(self):
        """
        """
        self.label_neg_count = lambda label_col: lambda x: \
            (x[label_col] == 0).sum()
        self.label_pos_count = lambda label_col: lambda x: \
            (x[label_col] == 1).sum()
        self.group_functions = self._get_group_functions()

    @staticmethod
    def _get_group_functions():
        """
        Helper function to accumulate lambda functions used in bias metrics
        calculations.
        """

        divide = lambda x, y: x / y if y != 0 else pd.np.nan

        predicted_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] <= thres).sum()

        predicted_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] > thres).sum()

        predicted_pos_ratio_k = lambda rank_col, label_col, thres, k: lambda x: \
            divide((x[rank_col] <= thres).sum(), k + 0.0)

        predicted_pos_ratio_g = lambda rank_col, label_col, thres, k: lambda x: \
            divide((x[rank_col] <= thres).sum(), len(x) + 0.0)

        false_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 1)).sum()

        false_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 0)).sum()

        true_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 0)).sum()

        true_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 1)).sum()

        fpr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(),
                   (x[label_col] == 0).sum().astype(
                       float))

        tnr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum(), (x[label_col] ==
                                                                         0).sum().astype(
                float))

        fnr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(),
                   (x[label_col] == 1).sum().astype(
                       float))

        tpr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(), (x[label_col] ==
                                                                          1).sum().astype(
                float))

        fomr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(), (x[rank_col] >
                                                                         thres).sum(
            ).astype(float))

        npv = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum(),
                   (x[rank_col] > thres).sum().astype(
                       float))

        precision = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(), (x[rank_col] <=
                                                                          thres).sum(
            ).astype(float))

        fdr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(), (x[rank_col] <=
                                                                          thres).sum(
            ).astype(float))

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
                           'fp': false_pos_count,
                           'fn': false_neg_count,
                           'tn': true_neg_count,
                           'tp': true_pos_count}

        return group_functions

    def get_crosstabs(self, df, score_thresholds=None, model_id=1, attr_cols=None):
        """
        Creates univariate groups and calculates group metrics.

        :param df: a dataframe containing the following required columns [score,  label_value].
        :param score_thresholds: dictionary { 'rank_abs':[] , 'rank_pct':[], 'score':[] }
        :param model_id: the model ID on which to subset the df.
        :param attr_cols: optional, list of names of columns corresponding to
            group attributes (i.e., gender, age category, race, etc.).

        :return: A dataframe of group score, label, and error statistics and absolute bias metric values grouped by unique attribute values
        """
        if not attr_cols:
            non_attr_cols = ['id', 'model_id', 'entity_id', 'score', 'label_value', 'rank_abs', 'rank_pct']
            attr_cols = df.columns[~df.columns.isin(non_attr_cols)]  # index of the columns that are
        # check if all attr_cols exist in df
        check = [col in df.columns for col in attr_cols]
        if False in check:
            # todo: create separate check method that raises exception...
            logging.error('get_crosstabs: not all attribute columns provided exist in input dataframe!')
            # exit(1)
        # check if all columns are strings:
        non_string_cols = df.columns[(df.dtypes != object) & (df.dtypes != str) & (df.columns.isin(attr_cols))]
        if non_string_cols.empty is False:
            logging.error('get_crosstabs: input df was not preprocessed. There are non-string cols within attr_cols!')
            # exit(1)

        # if no score_thresholds are provided, we assume that rank_abs=number of 1s in the score column
        count_ones = None  # it also serves as flag to set parameter to 'binary'

        if not score_thresholds:
            df['score'] = df['score'].astype(float)
            count_ones = df['score'].value_counts().get(1.0, 0)
            score_thresholds = {'rank_abs': [count_ones]}

        print('model_id, score_thresholds', model_id, score_thresholds)
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
            col_group = df.fillna({col: 'pd.np.nan'}).groupby(col)
            counts = col_group.size()
            print('COUNTS:::', counts)
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
            #              rank_pct: [1.0, 5.0, 10.0]
            for thres_unit, thres_values in score_thresholds.items():

                for thres_val in thres_values:
                    flag = 0

                    # To discuss with Pedro: believe this might be the reason
                    # for cutoff error - if numbers are cumulative, per
                    # line 149 and line 150, why taking sum for k vs. max?
                    k = (df[thres_unit] <= thres_val).sum()

                    # denote threshold as binary if numeric count_ones value
                    # donate as [rank value]_abs or [rank_value]_pct otherwise
                    score_threshold = 'binary 0/1' if count_ones != None else str(thres_val) + '_' + thres_unit[-3:]
                    for name, func in self.group_functions.items():
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
                        # print(this_group_df.head(1))
                    dfs.append(this_group_df)
        # precision@	25_abs
        groups_df = pd.concat(dfs, ignore_index=True)
        priors_df = pd.concat(prior_dfs, ignore_index=True)
        groups_df = groups_df.merge(priors_df, on=['model_id', 'attribute_name',
                                                   'attribute_value'])
        return groups_df, attr_cols

    def list_absolute_metrics(self, df):
        """
        View list of all calculated absolute bias metrics in df
        """
        return df.columns.intersection(['fpr', 'fnr', 'tpr', 'tnr', 'for',
                                           'fdr', 'npv', 'precision', 'ppr',
                                           'pprev', 'prev'
                                        ]).tolist()
