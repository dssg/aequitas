import numpy as np
import pandas as pd


class Bias(object):

    def __init__(self, group_metrics={'PPR', 'PPrev', 'FDR', 'FOmR', 'FPR', 'FNR'},
                 fill_divbyzero=10.0):
        self.group_metrics = group_metrics
        self.fill_divbyzero = fill_divbyzero

    def get_bias_min_metric(self, df):
        """
            Calculates several ratios using the group metrics value and dividing by the minimum
            group metric value among all groups defined by each attribute
        :param df: the resulting dataframe from the group get_crosstabs
        :param group_metrics: the columns list corresponding to the group metrics for each
        we want to calculate the Disparity values
        :return: a dataframe
        """
        fill_1 = {}
        key_cols = ['model_id', 'parameter', 'group_variable']
        attr_groupby = df.groupby(key_cols)
        for group_metric in self.group_metrics:
            fill_1[group_metric] = 1.000000
            # creating disparity by dividing each group metric value by the corresponding min
            # value from the groups of the target attribute
            df[group_metric + ' Disparity'] = df[group_metric] / attr_groupby[
                group_metric].transform(min)
            df2 = pd.DataFrame()
            # but we also want to get the group_value of the reference group for each bias metric
            df2[key_cols + [group_metric + '_ref_group_value']] = df.loc[attr_groupby[
                group_metric].idxmin()][key_cols + ['group_value']]
            df = df.merge(df2, on=key_cols)
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, self.fill_divbyzero)
        df = df.fillna(value=fill_1)
        return df

    def get_bias_major_group(self, df):
        """

        :param df:
        :return:
        """
        fill_1 = {}
        key_cols = ['model_id', 'parameter', 'group_variable']
        attr_groupby = df.groupby(key_cols)
        for group_metric in self.group_metrics:
            fill_1[group_metric] = 1.000000
            df2 = pd.DataFrame()
            # but we also want to get the group_value of the reference group for each bias metric
            df2[key_cols + [group_metric + ' Disparity', group_metric + '_ref_group_value']] = \
            df.loc[
                attr_groupby['group_size'].idxmax()][key_cols + [group_metric, 'group_value']]
            df = df.merge(df2, on=key_cols)
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, self.fill_divbyzero)
        df = df.fillna(value=fill_1)
        return df

    def get_bias_predefined_group(self, df, ref_groups_list):
        """

        :param df:
        :param group_metrics_list:
        :param ref_groups_list:
        :return:
        """
        return df
