import logging
import numpy as np
import pandas as pd

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."


class Fairness(object):
    """
    """
    def __init__(self, fair_eval=None, tau=None, fair_measures_depend=None,
                 type_parity_depend=None, high_level_fairness_depend=None):
        """

        :param fair_eval: a lambda function that is used to assess fairness
            (e.g. 80% rule)
        :param tau: the threshold for fair/unfair evaluation
        :param fair_measures_depend: a dictionary containing fairness measures
            as keys and the corresponding input bias disparity as values
        :param type_parity_depend: a dictionary with Type I, Type II, and
            Equalized Odds fairness measures as keys and lists of their
            underlying bias metric parities as values
        :param high_level_fairness_depend: a dictionary with supervised and
            unsupervised fairness as keys and lists of their underlying metric
            parities as values.
        """

        if not fair_eval:
            self.fair_eval = lambda tau: lambda x: np.nan if np.isnan(x) else \
                (True if tau <= x <= 1 / tau else False)
        else:
            self.fair_eval = fair_eval
        # tau is the fairness_threshold and should be a real ]0.0 and 1.0]
        if not tau:
            self.tau = 0.8
        else:
            self.tau = tau

        # Set high-level fairness evaluation to NA (undefined) if both
        # underlying parity determinations are NA. If only one parity is NA,
        # evaluation is determined by the defined parity.
        self.high_level_pair_eval = lambda col1, col2: lambda x: np.nan if (np.isnan(x[col1]) and np.isnan(x[col2])) \
            else \
            (True if (x[col1] is True and x[col2] is True) else False)

        self.high_level_single_eval = lambda col: lambda x: np.nan if np.isnan(x[col]) else (True if x[col] is True else
        False)

        # the fair_measures_depend define the bias metrics that serve as input to the fairness evaluation and respective
        # fairness measure. basically these are the fairness measures supported by the current version of aequitas.

        if not fair_measures_depend:
            self.fair_measures_depend = {'Statistical Parity': 'ppr_disparity',
                                         'Impact Parity': 'pprev_disparity',
                                         'FDR Parity': 'fdr_disparity',
                                         'FPR Parity': 'fpr_disparity',
                                         'FOR Parity': 'for_disparity',
                                         'FNR Parity': 'fnr_disparity',
                                         'TPR Parity': 'tpr_disparity',
                                         'TNR Parity': 'tnr_disparity',
                                         'NPV Parity': 'npv_disparity',
                                         'Precision Parity': 'precision_disparity'
                                         }
        else:
            self.fair_measures_depend = fair_measures_depend

        # the self.fair_measures represents the list of fairness_measures to be calculated by default
        self.fair_measures_supported = self.fair_measures_depend.keys()

        if not type_parity_depend:
            self.type_parity_depend = {'TypeI Parity': ['FDR Parity', 'FPR Parity'],
                                       'TypeII Parity': ['FOR Parity', 'FNR Parity'],
                                       'Equalized Odds': ['FPR Parity', 'TPR Parity']}
        else:
            self.type_parity_depend = type_parity_depend

        # high level fairness_depend define which input fairness measures are used to calculate the high level ones
        if not high_level_fairness_depend:
            self.high_level_fairness_depend = {
                'Unsupervised Fairness': ['Statistical Parity', 'Impact Parity'],
                'Supervised Fairness': ['TypeI Parity', 'TypeII Parity']
            }
        else:
            self.high_level_fairness_depend = high_level_fairness_depend

    def get_fairness_measures_supported(self, input_df):
        """
        Determine fairness measures supported based on columns in data frame.
        """
        if 'label_value' not in input_df.columns:
            self.fair_measures_supported = ['Statistical Parity', 'Impact Parity']
        return self.fair_measures_supported


    def get_group_value_fairness(self, bias_df, tau=None, fair_measures_requested=None):
        """
        Calculates the fairness measures defined in fair_measures_requested
        dictionary and adds them as columns to the input bias_df.

        :param bias_df: the output dataframe from bias/ disparity calculation methods.
        :param tau: optional, the threshold for fair/ unfair evaluation.
        :param fair_measures_requested: optional, a dictionary containing fairness
            measures as keys and the corresponding input bias disparity as values.

        :return: Bias_df dataframe with additional columns for each
            of the fairness measures defined in the fair_measures dictionary
        """
        logging.info('get_group_value_fairness...')
        if not tau:
            tau = self.tau
        if not fair_measures_requested:
            fair_measures_requested = self.fair_measures_supported

        for fair, input in self.fair_measures_depend.items():
            if fair in fair_measures_requested:

                bias_df[fair] = bias_df[input].apply(self.fair_eval(tau))

        for fair, input in self.type_parity_depend.items():
            if input[0] in bias_df.columns:
                if input[1] in bias_df.columns:
                    bias_df[fair] = bias_df.apply(self.high_level_pair_eval(input[0], input[1]), axis=1)
                else:
                    bias_df[fair] = bias_df.apply(self.high_level_single_eval(input[0]), axis=1)
            elif input[1] in bias_df.columns:
                bias_df[fair] = bias_df.apply(self.high_level_single_eval(input[1]), axis=1)
            else:
                logging.warning('get_group_value_fairness: No Parity measure input found on bias_df')

        for fair, input in self.high_level_fairness_depend.items():
            if input[0] in bias_df.columns:
                if input[1] in bias_df.columns:
                    bias_df[fair] = bias_df.apply(self.high_level_pair_eval(input[0], input[1]), axis=1)
                else:
                    bias_df[fair] = bias_df.apply(self.high_level_single_eval(input[0]), axis=1)
            elif input[1] in bias_df.columns:
                bias_df[fair] = bias_df.apply(self.high_level_single_eval(input[1]), axis=1)
        if 'Unsupervised Fairness' not in bias_df.columns and 'Supervised Fairness' not in bias_df.columns:
            logging.info('get_group_value_fairness: No high level measure input found on bias_df' + input[1])
        return bias_df


    def _fill_groupby_attribute_fairness(self, groupby_obj, key_columns,
                                        group_attribute_df, measures):
        """
        Returns dataframe with values grouped by attribute_value
        """
        logging.info('fill_groupby_attribute_fairness')
        for key in measures:
            rows = []
            for group, values in groupby_obj:
                group_df = groupby_obj.get_group(group)
                if group_df[key].isnull().all():
                    row = group_df.iloc[0][key_columns + [key]]
                else:
                    group_df = group_df[group_df[key].notnull()][key_columns + [key]]
                    row = group_df.loc[group_df[key].astype(bool).idxmin()]
                rows.append(row)
            key_df = pd.DataFrame(rows)
            if group_attribute_df.empty:
                group_attribute_df = key_df
            else:
                group_attribute_df = group_attribute_df.merge(key_df, on=key_columns)
        return group_attribute_df


    def get_group_attribute_fairness(self, group_value_df, fair_measures_requested=None):
        """
        Determines whether the minimum value for each fairness measure in
        fair_measures_requested is 'False' across all attribute_values defined
        by a group attribute_name. If 'False' is present, determination for the
        attribute is False for given fairness measure.

        :param group_value_df: output dataframe of get_group_value_fairness() method
        :return: A dataframe of fairness measures at the attribute level (no attribute_values)
        """
        logging.info('get_group_attribute_fairness')
        if not fair_measures_requested:
            fair_measures_requested = self.fair_measures_supported
        group_attribute_df = pd.DataFrame()
        key_columns = ['model_id', 'score_threshold', 'attribute_name']
        groupby_variable = group_value_df.groupby(key_columns)
        # We need to do this because of NaNs. idxmin() on pandas raises keyerror if there is a NaN...
        group_attribute_df = self._fill_groupby_attribute_fairness(groupby_variable, key_columns, group_attribute_df,
                                                                  fair_measures_requested)
        if group_attribute_df.empty:
            raise Exception('get_group_attribute_fairness: no fairness measures requested found on input group_value_df columns')
        parity_cols = [col for col in self.type_parity_depend if col in group_value_df.columns]
        group_attribute_df = self._fill_groupby_attribute_fairness(groupby_variable, key_columns, group_attribute_df, parity_cols)
        highlevel_cols = [col for col in self.high_level_fairness_depend if col in group_value_df.columns]
        group_attribute_df = self._fill_groupby_attribute_fairness(groupby_variable, key_columns, group_attribute_df,
                                                                  highlevel_cols)
        return group_attribute_df


    def get_overall_fairness(self, group_attribute_df):
        """
        Calculates overall fairness regardless of the group_attributes.
        Searches for 'False' parity determinations across group_attributes and
        outputs 'True' determination if all group_attributes are fair.

        :param group_attribute_df: the output df of the get_group_attributes_fairness
        :return: A dictionary of overall, unsupervised, and supervised fairness determinations
        """
        overall_fairness = {}
        if 'Unsupervised Fairness' in group_attribute_df.columns:
            overall_fairness['Unsupervised Fairness'] = False if \
                group_attribute_df['Unsupervised Fairness'].min() == False else True

        if 'Supervised Fairness' in group_attribute_df.columns:
            overall_fairness['Supervised Fairness'] = False if group_attribute_df['Supervised Fairness'].min() == False else True

        fair_vals = [val for key, val in overall_fairness.items()]
        if False in fair_vals:
            overall_fairness['Overall Fairness'] = False
        elif True in fair_vals:
            overall_fairness['Overall Fairness'] = True
        else:
            overall_fairness['Overall Fairness'] = 'Undefined'
        return overall_fairness


    def list_parities(self, df):
        """
        View list of all parity determinations in df
        """
        all_fairness = self.type_parity_depend.keys() | \
                       self.high_level_fairness_depend.keys() | \
                       self.fair_measures_depend.keys()
        return list(all_fairness & set(df.columns))
