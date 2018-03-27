import pandas as pd


# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.

class Fairness(object):

    def __init__(self, fair_eval=None, tau=None, fair_measures=None, high_level_fairness=None):
        """

        :param fair_eval: a lambda function that is used to assess fairness (e.g. 80% rule)
        :param tau: the threshold for fair/unfair
        :param fair_measures: a dictionary containing fairness measures as keys and the
        corresponding input bias metric as values
        """
        if not fair_eval:
            self.fair_eval = lambda tau: lambda x: True if tau <= x <= 1 / tau else False
        else:
            self.fair_eval = fair_eval
        if not tau:
            self.tau = 0.8
        else:
            self.tau = tau
        if not fair_measures:
            self.fair_measures = {'Statistical Parity': 'ppr_disparity',
                                  'Impact Parity': 'pprev_disparity',
                                  'FDR Parity': 'fdr_disparity',
                                  'FPR Parity': 'fpr_disparity',
                                  'FOR Parity': 'for_disparity',
                                  'FNR Parity': 'fnr_disparity',
                                  'TypeI Parity': ['FDR Parity', 'FPR Parity'],
                                  'TypeII Parity': ['FOR Parity', 'FNR Parity']}
        if not high_level_fairness:
            self.high_level_fairness = {
                'Unsupervised Fairness': ['Statistical Parity', 'Impact Parity'],
                                  'Supervised Fairness': ['TypeI Parity', 'TypeII Parity']}

        else:
            self.fair_measures = fair_measures

        self.pair_eval = lambda col1, col2: lambda x: True if (x[col1] is True and x[col2] is
                                                               True) else False

    def get_group_value_fairness(self, bias_df, fair_eval=None, tau=None, fair_measures=None,
                                 high_level_fairness=None):
        """
            Calculates the fairness measures defined in the fair_measures dictionary and adds
            them as columns to the input bias_df

        :param bias_df: the output dataframe from the bias/disparities calculations
        :param fair_eval: (optional) see __init__()
        :param tau: (optional) see __init__()
        :param fair_measures: (optional) see __init__()
        :return: the input bias_df dataframe with additional columns for each of the fairness
        measures defined in the fair_measures dictionary
        """
        print('get_group_value_fairness')
        if not fair_eval:
            fair_eval = self.fair_eval
        if not tau:
            tau = self.tau
        if not fair_measures:
            fair_measures = self.fair_measures
        if not high_level_fairness:
            high_level_fairness = self.high_level_fairness
        for fair, bias in fair_measures.items():
            if type(bias) != list:
                bias_df[fair] = bias_df[bias].apply(fair_eval(tau))
        for fair, bias in fair_measures.items():
            if type(bias) == list:
                bias_df[fair] = bias_df.apply(self.pair_eval(bias[0], bias[1]), axis=1)
        for fair, bias in high_level_fairness.items():
            bias_df[fair] = bias_df.apply(self.pair_eval(bias[0], bias[1]), axis=1)
        return bias_df

    def get_group_variable_fairness(self, group_value_df, fair_measures=None,
                                    high_level_fairness=None):
        """

        :param group_value_df: the output dataframe of the get_group_value_fairness()
        :return: a new dataframe at the group_variable level (no group_values) with fairness
        measures at the group_variable level. Checks for the min (False) across the groups
        defined by the group_variable. IF the minimum is False then all group_variable is false
        for the given fairness measure.
        """
        print("get_group_variable_fairness")
        if not fair_measures:
            fair_measures = self.fair_measures
        if not high_level_fairness:
            high_level_fairness = self.high_level_fairness
        group_variable_df = pd.DataFrame()
        key_columns = ['model_id', 'parameter', 'group_variable']
        count = 0
        for key in fair_measures:
            df_min_idx = group_value_df.loc[group_value_df.groupby(key_columns)[key].idxmin()]
            if count == 0:
                group_variable_df[key_columns] = df_min_idx[key_columns]
            else:
                group_variable_df = group_variable_df.merge(df_min_idx[key_columns + [key]],
                                                            on=key_columns)
            count += 1
        for key in high_level_fairness:
            df_min_idx = group_value_df.loc[group_value_df.groupby(key_columns)[key].idxmin()]
            group_variable_df = group_variable_df.merge(df_min_idx[key_columns + [key]],
                                                        on=key_columns)

        return group_variable_df

    def get_overall_fairness(self, group_variable_df):
        """
            Calculates overall fairness regardless of the group_variables. It searches for
            unfairness across group_variables and outputs fairness if all group_variables are fair

        :param group_variable_df: the output df of the get_group_variable_fairness
        :return: dictionary with overall unsupervised/supervised fairness and fairness in general
        """
        overall_fairness = {}
        overall_fairness['Unsupervised Fairness'] = False if group_variable_df['Unsupervised ' \
                                                                               'Fairness'].min() \
                                                             == False else True
        overall_fairness['Supervised Fairness'] = False if group_variable_df['Supervised ' \
                                                                             'Fairness'].min() == \
                                                           False else True
        if overall_fairness['Unsupervised Fairness'] == True and \
                overall_fairness['Supervised Fairness'] == True:
            overall_fairness['Overall Fairness'] = True
        else:
            overall_fairness['Overall Fairness'] = False

        return overall_fairness

