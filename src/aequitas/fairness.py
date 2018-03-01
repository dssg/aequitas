class Fairness(object):

    def __init__(self, fair_eval=None, tau=None):
        if not fair_eval:
            self.fair_eval = lambda tau: lambda x: True if tau <= x <= 1 / tau else False
        else:
            self.fair_eval = fair_eval
        if not tau:
            self.tau = 0.8
        else:
            self.tau = tau
        self.pair_eval = lambda col1, col2: lambda x: True if (x[col1] is True and x[col2] is
                                                               True) else False

    def get_group_value_fairness(self, df, fair_eval=None, tau=None):
        if not fair_eval:
            fair_eval = self.fair_eval
        if not tau:
            tau = self.tau
        df['Statistical Parity'] = df['PPR_disparity'].apply(fair_eval(tau))
        df['Impact Parity'] = df['PPrev_disparity'].apply(fair_eval(tau))
        df['FDR Parity'] = df['FDR_disparity'].apply(fair_eval(tau))
        df['FPR Parity'] = df['FPR_disparity'].apply(fair_eval(tau))
        df['FOmR Parity'] = df['FOmR_disparity'].apply(fair_eval(tau))
        df['FNR Parity'] = df['FNR_disparity'].apply(fair_eval(tau))
        df['TypeI Parity'] = df.apply(self.pair_eval('FDR Parity', 'FPR Parity'), axis=1)
        df['TypeII Parity'] = df.apply(self.pair_eval('FOmR Parity', 'FNR Parity'), axis=1)
        df['Unsupervised Fairness'] = df.apply(self.pair_eval('Statistical Parity',
                                                              'Impact Parity'), axis=1)
        df['Supervised Fairness'] = df.apply(self.pair_eval('TypeI Parity', 'TypeII Parity'),
                                             axis=1)
        return df

    def get_group_variable_fairness(self, df):
        return df

    def get_overall_fairness(self, df):
        return df
