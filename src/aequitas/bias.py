import logging
from sys import exit

import numpy as np
import pandas as pd


class Bias(object):

    def __init__(self, key_columns=None, input_group_metrics=None, fill_divbyzero=None):
        """

        :param key_columns:
        :param input_group_metrics:
        :param fill_divbyzero:
        """
        if not key_columns:
            self.key_columns = ['model_id', 'parameter', 'group_variable']
        else:
            self.key_columns = key_columns
        if not input_group_metrics:
            self.input_group_metrics = ['PPR', 'PPrev', 'FDR', 'FOmR', 'FPR', 'FNR']
        else:
            self.input_group_metrics = input_group_metrics
        if not fill_divbyzero:
            self.fill_divbyzero = 10.00000
        else:
            self.fill_divbyzero = fill_divbyzero

    def get_disparity_min_metric(self, df, key_columns=None, input_group_metrics=None,
                                 fill_divbyzero=None):
        """
            Calculates several ratios using the group metrics value and dividing by the minimum
            group metric value among all groups defined by each attribute
        :param df: the resulting dataframe from the group get_crosstabs
        :param input_group_metrics: the columns list corresponding to the group metrics for each
        we want to calculate the Disparity values
        :return: a dataframe
        """

        print('get_disparity_min_metric()')
        if not key_columns:
            key_columns = self.key_columns
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero
        print('get_disparity_min_metric')
        fill_zeros = {}
        for group_metric in input_group_metrics:
            fill_zeros[group_metric + '_disparity'] = 1.000000
            try:
                # this groupby is being called every cycle. maybe we can create a list of df_groups
                # and merge df at the end? it can not be simply put outside the loop(the merge...)
                df_min_idx = df.loc[df.groupby(key_columns)[group_metric].idxmin()]
                # but we also want to get the group_value of the reference group for each bias metric
                df_to_merge = pd.DataFrame()
                df_to_merge[key_columns + [group_metric + '_disparity', group_metric +
                                           '_ref_group_value']] = \
                    df_min_idx[key_columns + [group_metric, 'group_value']]
            except KeyError:
                logging.error(
                    'get_bias_min_metric:: one of the following columns is not on the input '
                    'dataframe : model_id ,parameter,group_variable or any of the input_group_metrics '
                    'list')
                exit()
            df = df.merge(df_to_merge, on=key_columns)
            # creating disparity by dividing each group metric value by the corresponding min
            # value from the groups of the target attribute
            df[group_metric + '_disparity'] = df[group_metric] / df[group_metric + '_disparity']
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)
        df = df.fillna(value=fill_zeros)
        return df

    def get_disparity_major_group(self, df, key_columns=None, input_group_metrics=None,
                                  fill_divbyzero=None):
        """
            Calculates the bias (disparity) metrics for the predefined list of group metrics
            using the majority group within each attribute as the reference group (denominator)

        :param df: the returning dataframe from the group.get_crosstabs
        :return: a dataframe with the bias metrics as new columns and the ref group,
                it must have the same number of rows as the input dataframe
        """
        print('get_disparity_major_group()')
        if not key_columns:
            key_columns = self.key_columns
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero
        try:
            df_major_group = df.loc[df.groupby(key_columns)['group_size'].idxmax()]
        except KeyError:
            logging.error('get_bias_major_group:: one of the following columns is not on the input '
                          'dataframe : model_id ,parameter,group_variable, group_size ')
            exit()
        disparity_metrics = [col + '_disparity' for col in input_group_metrics]
        df_to_merge = pd.DataFrame()
        # we created the df_to_merge has a subset of the df_ref_group containing the target ref
        # group values which are now labeled as _disparity but we still need to perform the division
        df_to_merge[key_columns + disparity_metrics] = df_major_group[
            key_columns + input_group_metrics]
        # we now need to create the ref_group_value columns in the df_to_merge
        for col in input_group_metrics:
            df_to_merge[col + '_ref_group_value'] = df_major_group['group_value']
        df = df.merge(df_to_merge, on=key_columns)
        df[disparity_metrics] = df[input_group_metrics].divide(df[disparity_metrics].values)
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)
        # when there is a zero in the numerator and a zero in denominator it is considered NaN
        # after division, so if 0/0 we assume 1.0 disparity (they are the same...)
        fill_zeros = {metric: 1.000000 for metric in disparity_metrics}
        df = df.fillna(value=fill_zeros)
        return df

    def verify_ref_groups_dict_len(self, df, ref_groups_dict):
        if len(ref_groups_dict) != len(df['group_variable'].unique()):
            raise ValueError

    def verify_ref_group_loc(self, group_slice):
        if len(group_slice) < 1:
            raise ValueError

    def get_disparity_predefined_groups(self, df, ref_groups_dict, key_columns=None,
                                        input_group_metrics=None, fill_divbyzero=None):
        """
            Calculates the bias (disparity) metrics for the predefined list of input group metrics
            using a predefined reference group value for each attribute which is passed using
            ref_groups_dict (configs['reference_groups'])

        :param df: the output dataframe of the group.get_crosstabs
        :param ref_groups_dict: a dictionary {group_variable:group_value, ...}
        :param key_columns: optional, the key columns to use on joins
        :param input_group_metrics: optional, the group metrics to be used for creating the new
        disparity metrics
        :param fill_divbyzero: optional, fill value to use when divided by zero
        :return: a dataframe with same number of rows as the input but with additional
        disparity metrics columns and ref_group_values
        """
        print('get_disparity_predefined_group()')
        if not key_columns:
            key_columns = self.key_columns
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero
        try:
            self.verify_ref_groups_dict_len(df, ref_groups_dict)
        except ValueError:
            logging.error('Bias.get_disparity_predefined_groups(): the number of predefined group '
                          'values to use as reference is less than the actual number of '
                          'attributes in the input dataframe.')
            exit()
        df_ref_group = pd.DataFrame()
        try:
            for key, val in ref_groups_dict.items():
                group_slice = df.loc[(df['group_variable'] == key) & (df['group_value'] == val)]
                self.verify_ref_group_loc(group_slice)
                df_ref_group = pd.concat([df_ref_group, group_slice])
        except (KeyError, ValueError):
            logging.error('get_disparity_predefined_groups(): reference groups and values provided '
                          'do not exist as columns/values in the input dataframe.(Note: check for syntax errors)')
            exit()
        disparity_metrics = [col + '_disparity' for col in input_group_metrics]
        df_to_merge = pd.DataFrame()
        # we created the df_to_merge has a subset of the df_ref_group containing the target ref
        # group values which are now labeled as _disparity but we still need to perform the division
        df_to_merge[key_columns + disparity_metrics] = df_ref_group[
            key_columns + input_group_metrics]
        # we now need to create the ref_group_value columns in the df_to_merge
        for col in input_group_metrics:
            df_to_merge[col + '_ref_group_value'] = df_ref_group['group_value']
        df = df.merge(df_to_merge, on=key_columns)
        df[disparity_metrics] = df[input_group_metrics].divide(df[disparity_metrics].values)
        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)
        # when there is a zero in the numerator and a zero in denominator it is considered NaN
        # after division, so if 0/0 we assume 1.0 disparity (they are the same...)
        fill_zeros = {metric: 1.000000 for metric in disparity_metrics}
        df = df.fillna(value=fill_zeros)
        return df
