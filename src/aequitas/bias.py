import logging
from aequitas.plotting import assemble_ref_groups
import numpy as np
import pandas as pd
from scipy import stats

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

class Bias(object):
    """
    """
    default_key_columns = ('model_id', 'score_threshold', 'attribute_name')
    all_group_metrics = ('ppr', 'pprev', 'precision', 'fdr', 'for', 'fpr',
                         'fnr', 'tpr', 'tnr', 'npv')
    non_attr_cols = ('score', 'model_id', 'as_of_date', 'entity_id', 'rank_abs',
                     'rank_pct', 'id', 'label_value')

    def __init__(self, key_columns=default_key_columns, sample_df=None,
                 non_attr_cols=non_attr_cols,
                 input_group_metrics=all_group_metrics, fill_divbyzero=None):
        """

        :param key_columns: optional, key identifying columns for grouping
            variables and bias metrics in intermediate joins. Defaults are
            'model_id', 'score_threshold', 'attribute_name'.
        :param input_group_metrics: List of absolute bias metrics to calculate
        :param fill_divbyzero: optional, fill value to use when divided by
            zero. Default is None.
        """
        self.key_columns = list(key_columns)
        self.input_group_metrics = list(input_group_metrics)

        if not fill_divbyzero:
            self.fill_divbyzero = 10.00000
        else:
            self.fill_divbyzero = fill_divbyzero
        self.non_attr_cols = non_attr_cols
        self.significance_cols = set(self.input_group_metrics).union({'label_value', 'score'})

    def get_disparity_min_metric(self, df, original_df, key_columns=None,
                                 input_group_metrics=None, fill_divbyzero=None,
                                 check_significance=False,  alpha = 5e-2,
                                 mask_significance = True, label_score_ref='fpr',
                                 selected_significance=False):
        """
        Calculates disparities between groups for the predefined list of
        group metrics using the group with the minimum value for each absolute
        bias metric as the reference group (denominator).

        :param df: output dataframe of Group class get_crosstabs() method.
        :param original_df: a dataframe of sample features and model results.
            Includes a required 'score 'column and possible 'label_value' column.
        :param key_columns: optional, key identifying columns for grouping
            variables and bias metrics in intermediate joins. Defaults are
            'model_id', 'score_threshold', 'attribute_name'.
        :param input_group_metrics: optional, columns list corresponding to
            the group metrics for which we want to calculate disparity values
        :param fill_divbyzero: optional, fill value to use when divided by
            zero. Default is None.
        :param check_significance: whether to determine statistical significance
            for disparity metrics. Default is False.
        :param selected_significance: specific measures (beyond label_value and
            score) to which to limit statistical significance calculations when
            check_significance is True. Default is False, i.e., significance
            will be calculated for all metrics.
        :param alpha: statistical significance level to use in significance
            determination. Default is 5e-2 (0.05).
        :param mask_significance: whether to display a T/F mask over calculated
            p-values from statistical significance determination. Default is True.
        :param label_score_ref: default reference group to use for score and
            label_value statistical significance calculations.
        :return: A dataframe with same number of rows as the input (crosstab)
            with additional disparity metrics columns and ref_group_values
            for each metric.
        """

        logging.info('get_disparity_min_metric()')
        # record df column order
        original_cols = df.columns

        if not key_columns:
            key_columns = original_cols.intersection(self.key_columns).tolist()
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero

        for group_metric in input_group_metrics:

            try:
                # this groupby is being called every cycle. maybe we can create
                # a list of df_groups and merge df at the end? it can not be
                # simply put outside the loop(the merge...)
                idxmin = df.groupby(key_columns)[group_metric].idxmin()

                # if entire column for a group metric is NaN, cast min value index
                # column to the same index as that of any other group for that attribute
                if any(np.isnan(val) for val in idxmin.values):
                    if len(idxmin) >= 1:
                        idxmin_not_nan = df.reset_index().groupby(key_columns)['index'].min()
                        idxmin.loc[idxmin.isna()] = pd.merge(
                            left=idxmin.loc[idxmin.isna()], right=idxmin_not_nan,
                            left_index=True, right_index=True, how='inner',
                        )['index'].values
                    else:
                        raise Exception(f"A minimum value for group_metric "
                                      f"{group_metric} could not be calculated.")

                df_min_idx = df.loc[idxmin]

                # but we also want to get the group_value of the reference group
                # for each bias metric
                df_to_merge = pd.DataFrame()
                df_to_merge[key_columns + [group_metric + '_disparity', group_metric +
                                           '_ref_group_value']] = \
                    df_min_idx[key_columns + [group_metric, 'attribute_value']]

            except KeyError:
                raise KeyError(
                    'get_bias_min_metric:: one of the following columns is not '
                    'on the input dataframe : model_id, parameter, attribute_name '
                    'or any of the input_group_metrics '
                    'list')

            df = df.merge(df_to_merge, on=key_columns)
            # creating disparity by dividing each group metric value by the
            # corresponding min value from the groups of the target attribute
            df[group_metric + '_disparity'] = \
                df[group_metric] / df[group_metric + '_disparity']
            # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)

        if not check_significance:
            return df

        else:
            # add statistical_significance
            if not selected_significance:
                selected_significance = self.input_group_metrics

                # only proceed with columns actually in dataframe
            selected_significance = set(original_cols.intersection(selected_significance))

            # always includes label and score significance
            selected_significance = selected_significance.union({'label_value', 'score'})

            ref_groups_dict = assemble_ref_groups(df, ref_group_flag='_ref_group_value',
                                                  specific_measures=selected_significance,
                                                  label_score_ref=label_score_ref)

            attr_cols = df['attribute_name'].unique()
            # run significance method on bias-augmented crosstab based on false
            # positives, false negatives, scores, and label values in original df
            self._get_statistical_significance(
                original_df, df, ref_dict=ref_groups_dict, score_thresholds=None,
                attr_cols=attr_cols, alpha=5e-2, selected_significance=selected_significance)

            # if specified, apply T/F mask to significance columns
            if mask_significance:
                significance_cols = df.columns[df.columns.str.contains('_significance')]
                truemask = df.loc[:, significance_cols] < alpha
                falsemask = df.loc[:, significance_cols] >= alpha

                df.loc[:, significance_cols] = np.select(
                    [truemask, falsemask], [True, False], default=None)

            # order new calculated metric columns: disparity, ref_group, then
            # significance for each
            base_sig=['label_value_significance', 'score_significance']

            new_cols = sorted(
                list(set(df.columns) - set(original_cols) - set(base_sig))
            )

            return df[original_cols.tolist() + base_sig + new_cols]


    def get_disparity_major_group(self, df, original_df, key_columns=None,
                                  input_group_metrics=None,
                                  fill_divbyzero=None, check_significance=False,
                                  alpha = 5e-2, mask_significance=True,
                                  selected_significance=False):
        """
        Calculates disparities between groups for the predefined list of group
        metrics using the majority group within each attribute as the reference
        group (denominator).

        :param df: output dataframe of Group class get_crosstabs() method.
        :param original_df: a dataframe of sample features and model results.
            Includes a required 'score 'column and possible 'label_value' column.
        :param key_columns: optional, key identifying columns for grouping
            variables and bias metrics in intermediate joins. Defaults are
            'model_id', 'score_threshold', 'attribute_name'.
        :param input_group_metrics: optional, columns list corresponding to
            the group metrics for which we want to calculate disparity values
        :param fill_divbyzero: optional, fill value to use when divided by
            zero. Default is None.
        :param check_significance: whether to determine statistical signifance
            for disparity metrics. Default is False.
        :param selected_significance: specific measures (beyond label_value and
            score) to which to limit statistical significance calculations when
            check_significance is True. Default is False, i.e., significance
            will be calculated for all metrics.
        :param alpha: statistical significance level to use in significance
            determination. Default is 5e-2 (0.05).
        :param mask_significance: whether to display a T/F mask over calculated
            p-values from statistical significance determination. Default is True.
        :return: A dataframe with same number of rows as the input (crosstab)
            with additional disparity metrics columns and ref_group_values
            for each metric.
        """
        logging.info('get_disparity_major_group()')
        # record df column order
        original_cols = df.columns

        if not key_columns:
            key_columns = original_cols.intersection(self.key_columns).tolist()
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero

        try:
            df_major_group = df.loc[df.groupby(key_columns)['group_size'].idxmax()]
        except KeyError:
            raise KeyError('get_bias_major_group:: one of the following columns '
                          'is not on the input dataframe : model_id, parameter, '
                          'attribute_name, group_size')

        disparity_metrics = [col + '_disparity' for col in input_group_metrics]
        df_to_merge = pd.DataFrame()

        # we created the df_to_merge has a subset of the df_ref_group containing
        # the target ref group values which are now labeled as _disparity but
        # we still need to perform the division
        df_to_merge[key_columns + disparity_metrics] = df_major_group[
            key_columns + input_group_metrics]

        # we now need to create the ref_group_value columns in the df_to_merge
        for col in input_group_metrics:
            df_to_merge[col + '_ref_group_value'] = df_major_group['attribute_value']
        df = df.merge(df_to_merge, on=key_columns)
        df[disparity_metrics] = df[input_group_metrics].divide(df[disparity_metrics].values)

        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)

        # when there is a zero in the numerator and a zero in denominator it is
        # considered NaN after division, so if 0/0 we assume 1.0 disparity
        # (they are the same...)

        # default is to use the same ref groups as df, need to add functionality to
        # compile ref_groups_dict based on a passed ref group for a given measure
        if not check_significance:
            return df

        else:
            if not selected_significance:
                selected_significance = self.input_group_metrics

            # only proceed with columns actually in dataframe
            selected_significance = set(original_cols.intersection(selected_significance))

            # always includes label and score significance
            selected_significance = selected_significance.union({'label_value', 'score'})

            ref_groups_dict = assemble_ref_groups(df, ref_group_flag='_ref_group_value',
                                                  specific_measures=selected_significance,
                                                  label_score_ref=None)

            ref_groups_dict = assemble_ref_groups(df, ref_group_flag='_ref_group_value',
                                                  specific_measures=selected_significance,
                                                  label_score_ref=None)

            attr_cols = df['attribute_name'].unique()

            for attribute in attr_cols:
                largest_group = df_major_group.loc[df_major_group['attribute_name'] == attribute,
                                                   'attribute_value'].values.tolist()[0]
                ref_groups_dict[attribute]['label_value'] = largest_group
                ref_groups_dict[attribute]['score'] = largest_group


            # run significance method on bias-augmented crosstab based on false
            # positives, false negatives, scores, and label values in original df
            self._get_statistical_significance(
                original_df, df, ref_dict=ref_groups_dict, score_thresholds=None,
                attr_cols=attr_cols, alpha=5e-2, selected_significance=selected_significance)

            # if specified, apply T/F mask to significance columns
            if mask_significance:
                significance_cols = df.columns[df.columns.str.contains('_significance')]
                truemask = df.loc[:, significance_cols] < alpha
                falsemask = df.loc[:, significance_cols] >= alpha

                df.loc[:, significance_cols] = np.select(
                    [truemask, falsemask], [True, False], default=None)

            # check what new disparity columns are and order as disparity,
            # ref_group, significance for each
            base_sig=['label_value_significance', 'score_significance']

            new_cols = sorted(
                list(set(df.columns) - set(original_cols) - set(base_sig))
            )

            return df[original_cols.tolist() + base_sig + new_cols]


    def _verify_ref_groups_dict_len(self, df, ref_groups_dict):
        if len(ref_groups_dict) != len(df['attribute_name'].unique()):
            raise ValueError

    def _verify_ref_group_loc(self, group_slice):
        if len(group_slice) < 1:
            raise ValueError

    def get_disparity_predefined_groups(self, df, original_df, ref_groups_dict,
                                        key_columns=None,
                                        input_group_metrics=None,
                                        fill_divbyzero=None,
                                        check_significance=False, alpha=5e-2,
                                        mask_significance=True,
                                        selected_significance=False):
        """
        Calculates disparities between groups for the predefined list of group
        metrics using a predefined reference group (denominator) value for each
        attribute.

        :param df: output dataframe of Group class get_crosstabs() method.
        :param original_df: dataframe of sample features and model results.
            Includes a required 'score 'column and possible 'label_value' column.
        :param ref_groups_dict: dictionary of format: {'attribute_name': 'attribute_value', ...}
        :param key_columns: optional, key identifying columns for grouping
            variables and bias metrics in intermediate joins. Defaults are
            'model_id', 'score_threshold', 'attribute_name'.
        :param input_group_metrics: optional, columns list corresponding to
            the group metrics for which we want to calculate disparity values
        :param fill_divbyzero: optional, fill value to use when divided by
            zero. Default is None.
        :param check_significance: whether to determine statistical signifance
            for disparity metrics. Default is False.
        :param selected_significance: specific measures (beyond label_value and
            score) to which to limit statistical significance calculations when
            check_significance is True. Default is False, i.e., significance
            will be calculated for all metrics.
        :param alpha: statistical significance level to use in significance
            determination. Default is 5e-2 (0.05).
        :param mask_significance: whether to display a T/F mask over calculated
            p-values from statistical significance determination. Default is True.
        :return: A dataframe with same number of rows as the input (crosstab)
            with additional disparity metrics columns and ref_group_values
            for each metric.
        """
        logging.info('get_disparity_predefined_group()')
        # record df column order
        original_cols = df.columns

        if not key_columns:
            key_columns = original_cols.intersection(self.key_columns).tolist()
        if not input_group_metrics:
            input_group_metrics = self.input_group_metrics
        if not fill_divbyzero:
            fill_divbyzero = self.fill_divbyzero

        try:
            self._verify_ref_groups_dict_len(df, ref_groups_dict)
        except ValueError:
            raise ValueError('Bias.get_disparity_predefined_groups(): the number of '
                          'predefined group values to use as reference is less '
                          'than the actual number of attributes in the input '
                          'dataframe.')

        df_ref_group = pd.DataFrame()
        try:
            for key, val in ref_groups_dict.items():
                group_slice = df.loc[(df['attribute_name'] == key) &
                                     (df['attribute_value'] == val)]
                self._verify_ref_group_loc(group_slice)
                df_ref_group = pd.concat([df_ref_group, group_slice])
        except KeyError:
            raise KeyError('get_disparity_predefined_groups(): reference groups '
                          'and values provided do not exist as columns/values '
                          'in the input dataframe.(Note: check for syntax errors)')
        except ValueError:
            raise ValueError('get_disparity_predefined_groups(): reference groups '
                           'and values provided do not exist as columns/values '
                           'in the input dataframe.(Note: check for syntax errors)')

        disparity_metrics = [col + '_disparity' for col in input_group_metrics]
        df_to_merge = pd.DataFrame()

        # we created the df_to_merge has a subset of the df_ref_group containing
        # the target ref group values which are now labeled as _disparity but
        # we still need to perform the division
        df_to_merge[key_columns + disparity_metrics] = df_ref_group[
            key_columns + input_group_metrics]

        # we now need to create the ref_group_value columns in the df_to_merge
        for col in input_group_metrics:
            df_to_merge[col + '_ref_group_value'] = df_ref_group['attribute_value']
        df = df.merge(df_to_merge, on=key_columns)
        df[disparity_metrics] = df[input_group_metrics].divide(df[disparity_metrics].values)

        # We are capping the disparity values to 10.0 when divided by zero...
        df = df.replace(np.inf, fill_divbyzero)
        if not check_significance:
            return df

        else:
            if not selected_significance:
                selected_significance = self.input_group_metrics

            # only proceed with columns actually in dataframe
            selected_significance = set( original_cols.intersection(selected_significance) )

            # always includes label and score significance
            selected_significance = selected_significance.union({'label_value', 'score'})

            # compile dictionary of reference groups based on bias-augmented crosstab
            full_ref_dict = {}

            # for predefined groups, use the largest of the predefined groups as
            # ref group for score and label value
            # key is an attribute_name, val is an attribute_value
            for key, val in ref_groups_dict.items():
                full_ref_dict[key] = {'label_value': val,
                                      'score': val}
                for measure in selected_significance:
                    full_ref_dict[key][measure] = val

            # run significance method on bias-augmented crosstab based on false
            # positives, false negatives, scores, and label values in original df
            self._get_statistical_significance(
                original_df, df, ref_dict=full_ref_dict, score_thresholds=None,
                attr_cols=None, alpha=5e-2, selected_significance=selected_significance)

            # if specified, apply T/F mask to significance columns
            if mask_significance:
                significance_cols = df.columns[df.columns.str.contains('_significance')]
                truemask = df.loc[:, significance_cols] < alpha
                falsemask = df.loc[:, significance_cols] >= alpha

                df.loc[:, significance_cols] = np.select(
                    [truemask, falsemask], [True, False], default=None)

            # check what new disparity columns are and order as disparity,
            # ref_group, significance for each
            base_sig=['label_value_significance', 'score_significance']

            new_cols = sorted(
                list(set(df.columns) - set(original_cols) - set(base_sig))
            )

            return df[original_cols.tolist() + base_sig + new_cols]



    @staticmethod
    def _get_measure_sample(original_df, attribute, measure):
        """
        Helper function for _get_statistical_significance() (via
        _calculate_significance() function). Convert dataframe to samples for
        given attribute group.

        :param original_df: a dataframe containing a required raw 'score' column
            and possible raw 'label_value' column.
        :param attribute: Attribute of interest in dataset (ex: race, sex, age
            category)
        :param measure: Metric of interest for which to calculate significance
            (false positives, false negatives, score, label_value)

        :return: A dictionary of binary 'samples' for each attribute group
        """
        return original_df.groupby(attribute).apply(
            lambda f: f.loc[f[measure].notnull(), measure].values.tolist()).to_dict()


    @staticmethod
    def _check_equal_variance(sample_dict, ref_group, alpha=5e-2):
        """
        Helper function for _get_statistical_significance() (via
        _calculate_significance() function).

        :param sample_dict: dictionary of binary samples for equal variance
            comparison.
        :param ref_group: Group to use as reference group for statistical
            significance calculation.
        :param alpha: Level at which to determine statistical significance.
            Default is 5e-2 (0.05).

        :return: Dictionary indicating whether each group has equal variance
        (in comparison with reference group)
        """

        # Immediately set ref_group status for equal variance (with itself): True

        eq_variance = {ref_group: True}

        for attr_value, sample in sample_dict.items():
            # make default normality_p value (only used when len(sample) < 20)
            # large enough that it is always greater than alpha
            normality_p = np.inf

            # skew test requires at least 20 samples
            if len(sample) > 20:
                _, normality_p = stats.normaltest(sample, axis=None, nan_policy='omit')

            # if tested normality is False or less than 8 samples, use levene
            # test to check equal variance between groups
            if normality_p < alpha or len(sample) <= 20:
                # if ref_group is not normal, can't use f-test or bartlett test
                # for any samples, so check for equal variance against ref_group
                # using levene test for all groups and return dict
                # (since includes all groups)
                if attr_value == ref_group:
                    for group, sample_list in sample_dict.items():
                        _, equal_variance_p = stats.levene(sample_dict[ref_group],
                                                           sample_list,
                                                           center='median')

                        eq_variance[group] = equal_variance_p >= alpha


                    return eq_variance

                # if a non-ref group is not normal, can't use f-test or bartlett
                # for that group, check for equal variance (against ref_group)
                # using levene test and add result to dictionary
                _, equal_variance_p = stats.levene(
                    sample_dict[ref_group], sample, center='median')

                eq_variance[attr_value] = equal_variance_p >= alpha

        # for all normally distributed non-ref groups, use bartlett test to
        # check for equal variance (against ref_group). Add results to dict
        untested_groups = sample_dict.keys() - eq_variance.keys() - set(ref_group)
        untested = {key: val for (key, val) in sample_dict.items()
                    if key in untested_groups}
        for attr_value, sample_list in untested.items():
            _, equal_variance_p = stats.bartlett(sample_dict[ref_group], sample_list)

            eq_variance[attr_value] = equal_variance_p >= alpha

        return eq_variance


    @classmethod
    def _calculate_significance(cls, original_df, disparity_df, attribute,
                               measure, ref_dict, alpha=5e-2):
        """
        Helper function for _get_statistical_significance. Pulls samples from
        original df, checks for equal variance between population groups and
        reference group, runs t-test between groups and reference group, and
        adds p-values to disparity_df for a given measure (ex: false positives)

        :param original_df: a dataframe containing a required raw 'score' column
            and possible raw 'label_value' column.
        :param disparity_df: Expansion of get_crosstabs() output with additional
            columns for disparity metrics and each metric's reference group to
            which significance must be added
        :param attribute: Attribute for which to calculate statistical
            significance of a measure
        :param measure: Measure for which to calculate statistical significance
        :param ref_dict: Dictionary indicating reference group for each
            attribute/ measure combination
        :param alpha: Level at which to determine statistical significance.
            Default is 5e-2 (0.05).

        :return: dataframe with same number of rows as the input (crosstab)
            but with additional disparity metrics columns, ref_group_values, and
            statistical significance (p-values) of specific metrics.

        """
        binaries_lookup = {'label_value': 'label_value', 'binary_fpr': 'fpr',
                           'binary_tpr': 'tpr', 'binary_tnr': 'tnr',
                           'binary_fnr': 'fnr', 'binary_score': 'score',
                           'binary_precision': 'precision', 'binary_npv': 'npv',
                           'binary_for': 'for', 'binary_fdr': 'fdr',
                           'binary_ppr': 'ppr', 'binary_pprev': 'pprev'
                            }

        ref_group = ref_dict[attribute][binaries_lookup.get(measure)]

        # create dictionary of "samples" (binary values for false positive,
        # false negative, label value, score) based on original data frame
        sample_dict = cls._get_measure_sample(original_df=original_df,
                                              attribute=attribute, measure=measure)


        # run SciPy equal variance tests between each group and a given
        # reference group, store results in dictionary to pass to statistical
        # significance tests
        eq_variance_dict = cls._check_equal_variance(sample_dict=sample_dict,
                                                    ref_group=ref_group,
                                                    alpha=alpha)
        # run SciPy statistical significance test between each group and
        # reference group
        for attr_val, eq_var in sample_dict.items():

            _, difference_significance_p = stats.ttest_ind(
                sample_dict[ref_group],
                sample_dict[attr_val],
                axis=None,
                equal_var=eq_variance_dict[attr_val],
                nan_policy='omit')

            measure = measure.replace('binary_', '')

            # add column to crosstab to indicate statistical significance
            disparity_df.loc[disparity_df['attribute_value'] == attr_val,
                             measure + '_significance'] = difference_significance_p

        return disparity_df



    @classmethod
    def _get_statistical_significance(cls, original_df, disparity_df, ref_dict,
                                     score_thresholds=None,
                                     attr_cols=None, alpha=5e-2,
                                     selected_significance=False):
        """

        :param original_df: a dataframe containing a required raw 'score' column
            and possible raw 'label_value' column.
        :param disparity_df: Expansion of get_crosstabs() output with additional
            columns for disparity metrics and each metric's reference group to
            which significance must be added
        :param ref_dict: Dictionary indicating reference group for each
            attribute/ measure combination
        :param score_thresholds: a dictionary { 'rank_abs':[] , 'rank_pct':[], 'score':[] }
        :param model_id: (Future functionality) ID(s) of models for which to check
            statistical significance
        :param attr_cols: Columns indicating attribute values in original_df
        :param alpha: Level at which to determine statistical significance.
            Default is 5e-2 (0.05).

        :return: dataframe with same number of rows as the input (crosstab)
            but with additional disparity metrics columns, ref_group_values, and
            statistical significance (p-values) of specific metrics.
        """
        if 'label_value' not in original_df.columns:
            raise ValueError(
                "Column 'label_value' not in dataframe. Label values are "
                "required for computing statistical significance of supervised "
                "metrics.")

        if attr_cols is None:
            non_attr_cols = [
                'id', 'model_id', 'entity_id', 'score', 'label_value',
                'rank_abs', 'rank_pct']

            # index of the columns that are attributes
            attr_cols = original_df.columns[~original_df.columns.isin(non_attr_cols)]

        # check if all attr_cols exist in df
        if set(attr_cols) - set(original_df.columns):
            raise ValueError(
                f"Not all attribute columns provided '{attr_cols}' exist in "
                f"input dataframe.")

        # check if all columns are strings:
        non_string_cols = \
            original_df.columns[
                (original_df.dtypes != object) &
                (original_df.dtypes != str) &
                (original_df.columns.isin(attr_cols))]

        binary_inclusions = {f'binary_{col}' for col in (selected_significance or
                                                         cls.significance_cols)
                             if col != 'label_value'}

        # check whether necessary binary columns already created
        if not non_string_cols.empty:
            # check whether error message is required based on if only binary columns
            if not all('binary_' in col for col in non_string_cols):
                raise ValueError(
                    'get_statistical_significance: statistical significance was '
                    'not calculated. There are non-string cols within attr_cols.')


            # skip binary column creation if all necessary columns already created
            elif not binary_inclusions - set(non_string_cols):
                measures = set(original_df.columns[original_df.columns.str.contains('binary_')])
                measures = measures.union({'label_value'})

                for attribute in set(attr_cols) - measures:
                    for measure in measures:

                        # only calculate significance if in selected_significance
                        if (measure in selected_significance) or (
                                measure.replace('binary_', '') in selected_significance):
                            cls._calculate_significance(original_df, disparity_df,
                                                        attribute, measure, ref_dict=ref_dict,
                                                        alpha=alpha)

                return disparity_df

            else:
                # drop old binary columns and generate those in binary_inclusions
                # as though non_string_cols was empty
                drop_cols = list(original_df.columns[original_df.columns.str.contains('binary_')])
                drop_cols += list(original_df.columns[original_df.columns.str.contains('rank_')])
                original_df = original_df.drop(drop_cols, axis=1)

        # create binary columns if non_string_cols is empty or binaries dropped

        # if no score_thresholds are provided, we assume that rank_abs equals
        # the number  of 1s in the score column; it also serves as flag to set
        # parameter to 'binary'
        count_ones = None
        if not score_thresholds:
            original_df.loc[:, 'score'] = original_df.loc[:,'score'].astype(float)

            count_ones = original_df['score'].value_counts().get(1.0, 0)
            score_thresholds = {'rank_abs': [count_ones]}

        original_df.sort_values('score', ascending=False, inplace=True)
        original_df.loc[:,'rank_abs'] = range(1, len(original_df) + 1)
        original_df.loc[:,'rank_pct'] = original_df.loc[:,'rank_abs'] / len(original_df)

        # Define formula for binary false positive, false negative, binary score
        binary_false_pos = lambda rank_col, label_col, thres: lambda x: (
            (x[rank_col] <= thres) & (x[label_col] == 0)).astype(int)

        binary_false_neg = lambda rank_col, label_col, thres: lambda x: (
            (x[rank_col] > thres) & (x[label_col] == 1)).astype(int)

        binary_score = lambda rank_col, label_col, thres: lambda x: (
                x[rank_col] <= thres).astype(int)

        binary_col_functions = {'binary_score': binary_score,
                                'binary_fpr': binary_false_pos,
                                'binary_fnr': binary_false_neg
                                }

        for thres_unit, thres_values in score_thresholds.items():
            for thres_val in thres_values:
                for name, func in binary_col_functions.items():
                    func = func(thres_unit, 'label_value', thres_val)
                    original_df.loc[:, name] = original_df.apply(func, axis=1)

        # add columns for error-based significance
        # precision, tnr, fdr are based on false positives
        SIGNIF_BASES = {
            'binary_precision': 'binary_fpr',
            'binary_tnr': 'binary_fpr',
            'binary_fdr': 'binary_fpr',
            'binary_npv': 'binary_fnr',
            'binary_tpr': 'binary_fnr',
            'binary_for': 'binary_fnr',
            'binary_ppr': 'binary_score',
            'binary_pprev': 'binary_score'}

        # add columns for the rest of columns in dictionary keys
        # binary score, fnr, fpr already added above
        for col in (binary_inclusions - binary_col_functions.keys()):
            original_df.loc[:, col] = original_df.loc[:,SIGNIF_BASES[col]]

        # ensure only predicted positive values included in true positive
        # error based metrics, and only predicted negative values in false
        # positive error based metrics
        POSITIVE_ONLY = ['binary_fpr', 'binary_tnr', 'binary_precision', 'binary_fdr']
        NEGATIVE_ONLY = ['binary_fnr', 'binary_tpr', 'binary_npv', 'binary_for']

        FPR_BASED = [metric for metric in POSITIVE_ONLY if metric in binary_inclusions]
        FNR_BASED = [metric for metric in NEGATIVE_ONLY if metric in binary_inclusions]

        original_df.loc[original_df['binary_score'] == 0, FPR_BASED] = np.nan
        original_df.loc[original_df['binary_score'] == 1, FNR_BASED] = np.nan

        measures = set(original_df.columns[original_df.columns.str.contains('binary_')])
        measures = measures.union({'label_value'})

        for attribute in attr_cols:
            for measure in measures:

                # only calculate significance if in selected_significance
                if (measure in selected_significance) or (measure.replace('binary_', '') in selected_significance):

                    cls._calculate_significance(original_df, disparity_df,
                                                attribute, measure, ref_dict=ref_dict,
                                                alpha=alpha)
        return disparity_df


    def list_disparities(self, df):
        """
        View list of all calculated disparities in df
        """
        try:
            return list(df.columns[df.columns.str.contains('_disparity')])
        except KeyError:
            raise KeyError("No disparity columns found in dataframe. Tip: "
                            "make sure you have already passed the dataframe "
                            "through a 'get_disparity_' method.")


    def list_significance(self, df):
        """
        View list of all calculated disparities in df
        """
        try:
            return list(df.columns[df.columns.str.contains('_significance')])
        except KeyError:
            raise KeyError("No significance columns found in dataframe. Tip: "
                            "make sure you set the 'check_significance' parameter"
                            " to True in 'get_disparity_' method(s).")


    def list_absolute_metrics(self, df):
        """
        View list of all calculated absolute bias metrics in df
        """
        return list(set(self.input_group_metrics).intersection(df.columns))
