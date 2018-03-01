import pandas as pd


class Fairness(object):

    def __init__(self, fair_eval=None, tau=None, fair_measures=None):
        if not fair_eval:
            self.fair_eval = lambda tau: lambda x: True if tau <= x <= 1 / tau else False
        else:
            self.fair_eval = fair_eval
        if not tau:
            self.tau = 0.8
        else:
            self.tau = tau
        if not fair_measures:
            self.fair_measures = {'Statistical Parity': 'PPR_disparity',
                                  'Impact Parity': 'PPrev_disparity',
                                  'FDR Parity': 'FDR_disparity',
                                  'FPR Parity': 'FPR_disparity',
                                  'FOmR Parity': 'FOmR_disparity',
                                  'FNR Parity': 'FNR_disparity',
                                  'TypeI Parity': ['FDR Parity', 'FPR Parity'],
                                  'TypeII Parity': ['FOmR Parity', 'FNR Parity'],
                                  'Unsupervised Fairness': ['Statistical Parity', 'Impact Parity'],
                                  'Supervised Fairness': ['TypeI Parity', 'TypeII Parity']}
        else:
            self.fair_measures = fair_measures

        self.pair_eval = lambda col1, col2: lambda x: True if (x[col1] is True and x[col2] is
                                                               True) else False

    def get_group_value_fairness(self, df, fair_eval=None, tau=None, fair_measures=None):
        if not fair_eval:
            fair_eval = self.fair_eval
        if not tau:
            tau = self.tau
        if not fair_measures:
            fair_measures = self.fair_measures

        for fair, bias in fair_measures.items():
            if type(bias) != list:
                df[fair] = df[bias].apply(fair_eval(tau))
        for fair, bias in fair_measures.items():
            if type(bias) == list:
                df[fair] = df.apply(self.pair_eval(bias[0], bias[1]), axis=1)

        return df

    def get_group_variable_fairness(self, df):
        fair_df = pd.DataFrame()
        key_columns = ['model_id', 'parameter', 'group_variable']
        attr_fair = df.groupby()['Statistical Parity'].agg(min)


        return df

    def get_overall_fairness(self, df):
        return df
