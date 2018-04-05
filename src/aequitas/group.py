import logging

import pandas as pd

from src.aequitas.preprocessing import preprocess_input_df

logging.getLogger(__name__)

# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#          Benedict Kuester
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved

class Group(object):
    def __init__(self):
        """

        """

        # the columns in the evaluation table and the thresholds we want to apply to them
        self.quantizers = {
            'quartiles': lambda s: 'quantile_' + pd.qcut(s, q=4, duplicates='drop', labels=False).
                map(lambda x: '%.0f' % x)}

        self.label_neg_count = lambda label_col: lambda x: \
            (x[label_col] == 0).sum()
        self.label_pos_count = lambda label_col: lambda x: \
            (x[label_col] == 1).sum()
        self.group_functions = self.get_group_functions()

    def get_group_functions(self):
        """

        :return:
        """

        divide = lambda x, y: x / y if y != 0 else 0.0

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

    def get_crosstabs(self, df, thresholds=None, model_id=1, non_attr_cols=None, preprocessed=False):
        """
        Creates univariate groups and calculates group metrics.

        :param df: a dataframe containing the following required columns [entity_id, as_of_date,
        model_id, score, rank_abs, rank_pct, label_value
        :param thresholds: a dictionary { 'rank_abs':[] , 'rank_pct':[], 'score':[] }
        :param model_id:
        :return:
        """
        if not non_attr_cols:
            non_attr_cols = ['id', 'model_id', 'entity_id', 'score', 'label_value', 'rank_abs', 'rank_pct']
        if not preprocessed:
            df, attr_cols = preprocess_input_df(df)
        else:
            attr_cols = df.columns[~df.columns.isin(non_attr_cols)]  # index of the columns that are
        # if no thresholds are provided, we assume that rank_abs=number of 1s in the score column
        if not thresholds:
            df['score'] = df['score'].astype(float)
            count_ones = df['score'].value_counts()[1.0]
            if count_ones == 0:
                logging.error('get_crosstabs: No threshold provided and there is no 1s in the score column.')
                exit(1)
            thresholds = {'rank_abs': [count_ones]}
        print('model_id, thresholds', model_id, thresholds)
        df = df.sort_values('score', ascending=False)
        df['rank_abs'] = range(1, len(df) + 1)
        df['rank_pct'] = df['rank_abs'] / len(df)
        dfs = []
        prior_dfs = []
        # calculate the bias for these columns
        # not default(non_attr_cols), therefore represent the group variables!
        print("Attribute Columns (Groups):", attr_cols.values)
        # for each group variable do
        for col in attr_cols:
            # find the priors_df
            col_group = df.fillna({col: 'nan'}).groupby(col)
            counts = col_group.entity_id.count()
            print('COUNTS:::', counts)
            # distinct entities within group value
            this_prior_df = pd.DataFrame({
                'model_id': [model_id] * len(counts),
                'group_variable': [col] * len(counts),
                'group_value': counts.index.values,
                'group_label_pos': col_group.apply(self.label_pos_count(
                    'label_value')).values,
                'group_label_neg': col_group.apply(self.label_neg_count(
                    'label_value')).values,
                'group_size': counts.values,
                'total_entities': [len(df)] * len(counts)
            })
            this_prior_df['prev'] = this_prior_df['group_label_pos'] / this_prior_df['group_size']
            # for each model_id and as_of_date the priors_df has length group_variables * group_values
            prior_dfs.append(this_prior_df)
            # we calculate the bias for two different types of thresholds (percentage ranks and absolute ranks)
            for thres_unit, thres_values in thresholds.items():
                for thres_val in thres_values:
                    flag = 0
                    k = (df[thres_unit] <= thres_val).sum()
                    for name, func in self.group_functions.items():
                        func = func(thres_unit, 'label_value', thres_val, k)
                        feat_bias = col_group.apply(func)
                        metrics_df = pd.DataFrame({
                            'model_id': [model_id] * len(feat_bias),
                            'parameter': [str(thres_val) + '_' + thres_unit[-3:]] * len(feat_bias),
                            'k': [k] * len(feat_bias),
                            'group_variable': [col] * len(feat_bias),
                            'group_value': feat_bias.index.values,
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
        groups_df = groups_df.merge(priors_df, on=['model_id', 'group_variable',
                                                   'group_value'])
        return groups_df, attr_cols


    def get_group_metrics(self, df):
        return 0
