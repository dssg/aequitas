import logging
from sys import exit

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

        print('get_bias_min_metric')
        fill_1 = {}
        key_cols = ['model_id', 'parameter', 'group_variable']
        for group_metric in self.group_metrics:
            fill_1[group_metric + ' Disparity'] = 1.000000
            # creating disparity by dividing each group metric value by the corresponding min
            # value from the groups of the target attribute
            df2 = pd.DataFrame()
            try:
                # this groupby is being called every cycle. maybe we can create a list of df_groups
                # and merge df at the end? it can not be simply put outside the loop(the merge...)
                df_group = df.loc[df.groupby(key_cols)[group_metric].idxmin()]
                # but we also want to get the group_value of the reference group for each bias metric
                df2[key_cols + [group_metric + ' Disparity', group_metric + '_ref_group_value']] = \
                    df_group[key_cols + [group_metric, 'group_value']]
            except KeyError:
                logging.error(
                    'get_bias_min_metric:: one of the following columns is not on the input '
                    'dataframe : model_id ,parameter,group_variable or any of the group_metrics '
                    'list')
                exit()
            df = df.merge(df2, on=key_cols)
            df[group_metric + ' Disparity'] = df[group_metric] / df[group_metric + ' Disparity']
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, self.fill_divbyzero)
        df = df.fillna(value=fill_1)
        return df

    def get_bias_major_group(self, df):
        """
            Calculates the bias (disparity) metrics for the predefined list of group metrics
            using the majority group within each attribute as the reference group (denominator)

        :param df: the returning dataframe from the group.get_crosstabs
        :return: a dataframe with the bias metrics as new columns and the ref group,
                it must have the same number of rows as the input dataframe
        """
        print('get_bias_major_group')
        fill_1 = {}
        key_cols = ['model_id', 'parameter', 'group_variable']
        try:
            df_group = df.loc[df.groupby(key_cols)['group_size'].idxmax()]
        except KeyError:
            logging.error('get_bias_major_group:: one of the following columns is not on the input '
                          'dataframe : model_id ,parameter,group_variable, group_size ')
            exit()
        count = 0
        for group_metric in self.group_metrics:
            count += 1
            print(count, group_metric)
            fill_1[group_metric + ' Disparity'] = 1.000000
            df2 = pd.DataFrame()
            # but we also want to get the group_value of the reference group for each bias metric
            # here we just getting the ref group metric value in the disparity column (it is not
            # the actual disparity value)
            df2[key_cols + [group_metric + ' Disparity', group_metric + '_ref_group_value']] = \
                df_group[key_cols + [group_metric, 'group_value']]
            df = df.merge(df2, on=key_cols)
            # now we are dividing the group metric value with the ref group metric value
            df[group_metric + ' Disparity'] = df[group_metric] / df[group_metric + ' Disparity']
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, self.fill_divbyzero)
        df = df.fillna(value=fill_1)
        return df

    def get_bias_predefined_group(self, df, ref_groups_dict):
        """

        :param df:
        :param group_metrics_list:
        :param ref_groups_list:
        :return:
        """

        #ref_groups_dict = {'race':'white','age':'18-25'}
        return df
