import datetime
import logging

import pandas as pd
from markdown2 import markdown
from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True

logging.getLogger(__name__)

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

#
#  DISCLAIMER: these methods were developed with a particular version of the webapp in mind. They lack flexibility (lots of
# hardcoded things)!!!
#             If new features(fairness measures, etc..) are added to the webapp this needs to change a lot...
#

def get_group_value_report(group_value_df):
    """

    :param group_value_df:
    :return:
    """
    group_value_report = {}
    the_false_df = group_value_df.loc[(group_value_df['Unsupervised Fairness'] == False) | (
            group_value_df['Supervised Fairness'] == False)]
    for index, row in the_false_df.iterrows():
        report = ''
        metrics = []
        group = row['group_variable'] + ' = ' + row['group_value']
        text1 = group + ' does not have '
        report += text1
        text2 = ''
        if row['Unsupervised Fairness'] is False:
            text2 += 'Unsupervised Fairness '
            text3 = ''
            if row['Statistical Parity'] is False:
                text3 += '(no Statistical Parity'
                ref_val = 0.0
                ref_group_value = group_value_df.loc[(group_value_df['group_variable'] == row[
                    'group_variable']) & (group_value_df['group_value'] == row[
                    'ppr_ref_group_value'])]['ppr'].values[0]
                ppr_text = '{:.0f}% of the group is selected, compared to {:.0f} % of the ' \
                           'reference group '.format(row['ppr'] * 100, ref_group_value * 100) + \
                           row['group_variable'] + ' = ' + row['ppr_ref_group_value']
                metrics.append(ppr_text)
            if row['Impact Parity'] is False:
                if text3 == '':
                    text3 += '(no Impact Parity)'
                else:
                    text3 += ', no Impact Parity)'
                pprev_text = ''
            else:
                text3 += ')'
            text2 += text3
        if row['Supervised Fairness'] is False:
            if text2 != '':
                text2 += ' neither '
            text2 += 'Supervised Fairness '
            text3 = ''
            if row['TypeI Parity'] is False:
                text3 += '(no Type I Parity'
            if row['TypeII Parity'] is False:
                if text3 == '':
                    text3 += '(no Type II Parity)'
                else:
                    text3 += ', no Type II Parity)'
            else:
                text3 += ') '
            text2 += text3
        report += text2
        group_value_report[group] = [report, metrics]

    return group_value_report


def get_highlevel_report(group_attribute_df):
    group_attribute_df = group_attribute_df.applymap(str)
    cols = ['attribute_name']
    if 'Unsupervised Fairness' in group_attribute_df.columns:
        cols.append('Unsupervised Fairness')
    if 'Supervised Fairness' in group_attribute_df.columns:
        cols.append('Supervised Fairness')
    group_attribute_df = group_attribute_df[cols]
    map = {}
    attr_list = group_attribute_df['attribute_name'].unique()
    for col in group_attribute_df.columns:
        if col == 'attribute_name':
            colstr = 'Attribute'
        else:
            colstr = col
        map[col] = colstr
        # to be able to click on true/false and redirect to the next section
        if col != 'attribute_name':
            for attr in attr_list:
                group_attribute_df.loc[group_attribute_df['attribute_name'] == attr, col] = '[' + group_attribute_df[col][
                    group_attribute_df['attribute_name'] == attr] + ']' + '(#' + '-'.join(attr.lower().split(' ')) + ')'
    for attr in attr_list:
        group_attribute_df = group_attribute_df.replace(attr, '[' + attr + ']' + '(#' + '-'.join(attr.lower().split(' ')) + ')')
    group_attribute_df = group_attribute_df.rename(index=str, columns=map)
    highlevel_report = tabulate(group_attribute_df, headers='keys', tablefmt='pipe', showindex='never', numalign="left")
    return highlevel_report


def get_parity_group_report(group_value_df, attribute, fairness_measures, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    def_cols = ['attribute_value']
    aux_df = group_value_df.loc[group_value_df['attribute_name'] == attribute]
    metrics = {}
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            metrics[par] = disp

    # getting a reference group label
    for col in aux_df.columns:
        if col in metrics.keys():
            ref_group = metrics[col].replace('_disparity', '_ref_group_value')
            idx = aux_df.loc[aux_df['attribute_value'] == aux_df[ref_group]].index
            aux_df.loc[idx, col] = 'Ref'

    map = {}
    aux_df = aux_df[def_cols + fairness_measures]
    for col in aux_df.columns:
        if col == 'attribute_value':
            colstr = 'Attribute Value'
        else:
            colstr = col
        map[col] = colstr  #+ ' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        aux_df[col] = '[' + aux_df[col] + ']' + '(#' + '-'.join(attribute.lower().split(' ')) + '-2)'
    aux_df = aux_df.rename(index=str, columns=map)
    cols_order = ['Attribute Value', 'Statistical Parity', 'Impact Parity', 'FDR Parity', 'FPR Parity', 'FOR Parity',
                  'FNR Parity']
    new_order = [col for col in cols_order if col in aux_df.columns]
    aux_df = aux_df[new_order]

    parity_group = tabulate(aux_df,
                            headers='keys',
                            tablefmt='pipe', showindex='never', numalign="left")
    return parity_group


def setup_group_value_df(group_value_df, fairness_measures, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    group_size = group_value_df['group_size_pct']
    metrics = {}
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            metrics[disp] = par
            metrics[disp.replace('_disparity', '')] = par
    aux_df = group_value_df[['attribute_name', 'attribute_value'] + list(metrics.keys())]
    for col in group_value_df.columns:
        if col in metrics.keys():
            # we want to keep the ref group without green/red so we need to know the name of the column to search for
            if not col.endswith('_disparity'):
                ref_group = col + '_ref_group_value'
            else:
                ref_group = col.replace('_disparity', '_ref_group_value')
            group_value_df.loc[(group_value_df[metrics[col]] == 'True') & (group_value_df['attribute_value'] != group_value_df[
                ref_group]), col] = '##green## ' + group_value_df[col][group_value_df[metrics[col]] == 'True']

            group_value_df.loc[group_value_df[metrics[col]] == 'False', col] = '##red##' + group_value_df[col][group_value_df[
                                                                                                                   metrics[
                                                                                                                       col]] == 'False']
    group_value_df['group_size_pct'] = group_size
    print('******************', group_value_df['group_size_pct'])
    return group_value_df


def get_disparities_group_report(group_value_df, attribute, fairness_measures, fairness_measures_depend):
    def_cols = ['attribute_value']
    metrics = {}
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            metrics[disp] = par
    aux_df = group_value_df.loc[group_value_df['attribute_name'] == attribute]
    aux_df = aux_df[def_cols + list(metrics.keys())]
    map = {}
    for col in aux_df.columns:
        colstr = col.replace('_', ' ')
        if col == 'attribute_value':
            colstr = 'Attribute Value'
        else:
            colstr = colstr.split(' ')[0].upper() + ' Disparity'
        map[col] = colstr  #+ ' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        aux_df[col] = '[' + aux_df[col] + ']' + '(#' + '-'.join(attribute.lower().split(' ')) + '-3)'
    aux_df = aux_df.rename(index=str, columns=map)
    # this is hardcoded. If metrics supported by aequitas change this needs to change
    cols_order = ['Attribute Value', 'PPR Disparity', 'PPREV Disparity', 'FDR Disparity', 'FPR Disparity', 'FOR Disparity',
                  'FNR Disparity']
    new_order = [col for col in cols_order if col in aux_df.columns]
    aux_df = aux_df[new_order]
    disparities_group = tabulate(aux_df,
                                 headers='keys',
                                 tablefmt='pipe', showindex='never', numalign="left")

    return disparities_group


def get_group_group_report(group_value_df, attribute, fairness_measures, fairness_measures_depend):
    # defining how to display stuff
    names = {'attribute_value': 'Attribute Value',
             'group_size_pct': 'Group Size Ratio'}
    def_cols = ['attribute_value', 'group_size_pct']
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            def_cols.append(disp.replace('_disparity', ''))
    aux_df = group_value_df.loc[group_value_df['attribute_name'] == attribute]
    aux_df = aux_df[def_cols]
    aux_df = aux_df.round(2)
    aux_df = aux_df.astype(str)
    # fixing the same order of columns every time!
    cols_order = ['attribute_value', 'group_size_pct', 'ppr', 'pprev', 'fdr', 'fpr', 'for', 'fnr']
    new_order = [col for col in cols_order if col in aux_df.columns]
    aux_df = aux_df[new_order]
    map = {}
    for col in aux_df.columns:
        if col in names:
            colstr = names[col]
        else:
            colstr = col.upper()
        map[col] = colstr  #+ ' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
    aux_df = aux_df.rename(index=str, columns=map)
    group_group = tabulate(aux_df,
                           headers='keys',
                           tablefmt='pipe', showindex='never', numalign="left")
    return group_group


def get_sentence_highlevel(fair_results):
    sent = '**The Bias Report audited the risk assessmentt system and has found that is has'
    is_fair = ' passed' if fair_results['Overall Fairness'] is True else ' failed'
    sent += is_fair + ' the audit with respect to the following fairness criteria:**\n\n'
    return sent



def get_statpar_text(group_value_df, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    fairness_metric = 'Statistical Parity'
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    false_dict = {}
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]
        sentence = ' is' \
                   ' **{group_metric_value}**% of positive class.' \
                   '' \
                   ''.format(
            group_metric_value='%.0f' % (float(row[group_metric]) * 100),
        )

        try:
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#equal-parity)' + sentence)
        except KeyError:
            false_dict[row['attribute_name']] = set()
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#equal-parity)' + sentence)

    if false_df.empty:
        cellref = '##green##Based on the fairness threshold used, the number of selected positives is similar across ' \
                  'different ' \
                  'groups.\n\n'
    else:
        cellref = ''
        for key in false_dict.keys():
            cellref += '**{attribute_name}:** ##br##&emsp;&emsp;&emsp;'.format(attribute_name=key)
            cellref += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##'

    return cellref


def get_impact_text(group_value_df, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    fairness_metric = 'Impact Parity'
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    false_dict = {}
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]

        sentence = ': **{group_metric_value}**% of the group is in the selected set (classified as positive),' \
                   ' in comparison to {ref_group_metric_value}% from the reference group \"{' \
                   'ref_group_value}\"' \
                   ''.format(
            group_metric_value='%.0f' % (float(row[group_metric]) * 100),
            attribute_value=row['attribute_value'],
            ref_group_metric_value='%.0f' % (float(ref_group_row[group_metric].values[0]) * 100),
            ref_group_value=row[ref_group_col])

        try:
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#proportional-parity)' + sentence)
        except KeyError:
            false_dict[row['attribute_name']] = set()
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#proportional-parity)' + sentence)

    if false_df.empty:
        cellref = '##green##Based on the fairness threshold used, the percentage of selected individuals from ' \
                  'each group is not disparate to the percentage of selected individuals of the respective reference group.\n\n'
    else:
        cellref = ''
        for key in false_dict.keys():
            cellref += '**{attribute_name}:** ##br##&emsp;&emsp;&emsp;'.format(attribute_name=key)
            cellref += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##'

    return cellref


def get_old_false_text(group_value_df, fairness_metric, fairness_measures_depend):
    names = {

        'fpr': 'false positive rate',
        'fnr': 'false negative rate',
        'fdr': 'false discovery rate',
        'for': 'false omission rate',
        'ppr': 'predicted positive ratio',
        'pprev': 'predicted prevalence (percentage of positives in the group)'
    }
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    false_dict = {}
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]
        sentence = ': **{bias_metric_value}**% ' \
                   'of the {group_metric_name} of the reference group \"{ref_group_value}\",' \
                   ' corresponding to a difference of {group_metric_value} vs {ref_group_metric_value}.' \
            .format(
            group_metric_name=names[group_metric],
            bias_metric_value='%.0f' % (float(row[bias_metric]) * 100),
            ref_group_value=row[ref_group_col],
            group_metric_value=row[group_metric],
            ref_group_metric_value=ref_group_row[group_metric].values[0])
        try:
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#false-positive-parity)' + sentence)
        except KeyError:
            false_dict[row['attribute_name']] = set()
            false_dict[row['attribute_name']].add('[' + row['attribute_value'] + '](#false-positive-parity)' + sentence)

    if false_df.empty:
        cellref = '##green##Based on the fairness threshold used, the percentage of selected elements from ' \
                  'each group is not disparate to the percentage of selected elements of the respective reference group.\n\n'
    else:
        cellref = ''
        for key in false_dict.keys():
            cellref += '**{attribute_name}:** ##br##&emsp;&emsp;&emsp;'.format(attribute_name=key)
            cellref += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##'

    return cellref


def get_false_text(group_value_df, fairness_metric, fairness_measures_depend):
    names = {

        'fpr': 'false positive rate',
        'fnr': 'false negative rate',
        'fdr': 'false discovery rate',
        'for': 'false omission rate',
        'ppr': 'predicted positive ratio',
        'pprev': 'predicted prevalence (percentage of positives in the group)'
    }
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    false_dict = {}
    ref_group_dict = {}
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]
        sentence = ' with ++span-red-init++{bias_metric_value}X++span-red-end++ Disparity'.format(
            bias_metric_value='%.2f' % float(row[bias_metric]))

        ref_group_dict[row['attribute_name']] = ' (with reference group as **' + row[ref_group_col] + '**)'


        sentence2 = '{group_metric_name} of this group is {group_metric_value} compared to {ref_group_metric_value} for the reference group {ref_group_value}.' \
            .format(
            group_metric_name=names[group_metric],
            ref_group_value=row[ref_group_col],
            group_metric_value=row[group_metric],
            ref_group_metric_value=ref_group_row[group_metric].values[0])

        try:
            false_dict[row['attribute_name']].add('##tooltip-start-title##' + sentence2 + '##tooltip-end-title##' + row[
                'attribute_value'] + '##tooltip-end-anchor##' +
                                                  sentence)
        except KeyError:
            false_dict[row['attribute_name']] = set()
            false_dict[row['attribute_name']].add('##tooltip-start-title##' + sentence2 + '##tooltip-end-title##' + row[
                'attribute_value'] + '##tooltip-end-anchor##' +
                                                  sentence)
    if false_df.empty:
        cellref = '++span-green-init++Based on the fairness threshold used, all groups passed the audit for this metric.++span-green-end++\n\n'
    else:
        cellref = ''
        for key in false_dict.keys():
            cellref += '**For {attribute_name}**'.format(attribute_name=key) + ref_group_dict[key] + '##br##&emsp;&emsp;&emsp;'
            cellref += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##  ##br##'

    return cellref


def get_highlevel_table(group_value_df, fairness_measures, ):
    supported_name = {'Statistical Parity': '[Equal Parity](#equal-parity)',
                      'Impact Parity': '[Proportional Parity](#proportional-parity)',
                      'TypeI Parity': '[False Positive Parity](#false-positive-parity)',
                      'TypeII Parity': '[False Negative Parity](#false-negative-parity)'}
    supported_outcome = {'Statistical Parity': 'Each group is represented equally.',
                         'Impact Parity': 'Each group is represented proportional to their representation in the overall population.',
                         'TypeI Parity': 'Each group has proportionally equal false positive errors made by the model.',
                         'TypeII Parity': 'Each group has proportionally equal false negative errors made by the model.'}
    supported_order = ['Statistical Parity', 'Impact Parity', 'TypeI Parity', 'TypeII Parity']
    # once again this is hardcoded because it's easy now, in the future make it mapped automatically

    map_ref_groups = {'Statistical Parity': ['ppr_ref_group_value'],
                      'Impact Parity': ['pprev_ref_group_value'],
                      'TypeI Parity': ['fpr_ref_group_value', 'fdr_ref_group_value'],
                      'TypeII Parity': ['fnr_ref_group_value', 'for_ref_group_value']}
    key_columns = ['model_id', 'score_threshold', 'attribute_name']
    fairness_measures_edited = []
    for meas in fairness_measures:
        if meas in ['FPR Parity', 'FDR Parity']:
            fairness_measures_edited.append('TypeI Parity')
        elif meas in ['FNR Parity', 'FOR Parity']:
            fairness_measures_edited.append('TypeII Parity')
        else:
            fairness_measures_edited.append(meas)
    fairness_measures_edited = set(fairness_measures_edited)
    raw = {
        'Fairness Criteria': [],
        'Desired Outcome': [],
        'Reference Groups Selected': [],
        'Unfairly Affected Groups': []
    }
    for measure in supported_order:
        if measure in fairness_measures_edited:
            raw['Fairness Criteria'].append(supported_name[measure])
            raw['Desired Outcome'].append(supported_outcome[measure])
            false_df = group_value_df.loc[group_value_df[measure] == False]
            ref_dict = {}
            false_dict = {}
            for index, row in false_df.iterrows():
                try:
                    false_dict[row['attribute_name']].add('[' + row['attribute_value'] + ']' + \
                                                          supported_name[measure][supported_name[measure].find('('):])
                except KeyError:
                    false_dict[row['attribute_name']] = set()
                    false_dict[row['attribute_name']].add('[' + row['attribute_value'] + ']' + \
                                                          supported_name[measure][supported_name[measure].find('('):])
            attr_order = []
            if len(false_dict) > 0:
                cell = ''
                attr_order = false_dict.keys()
                for key in attr_order:
                    cell += '**{attribute_name}:**'.format(attribute_name=key)
                    cell += '##br##&emsp;&emsp;&emsp;'
                    cell += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##'
                raw['Unfairly Affected Groups'].append(cell)
            else:
                if group_value_df[measure].isnull().all():
                    raw['Unfairly Affected Groups'].append('Undefined')
                else:
                    raw['Unfairly Affected Groups'].append('No Unfair Groups Found')

            for ref in map_ref_groups[measure]:
                groupby_refs = group_value_df.groupby(key_columns + [ref])
                for group, values in groupby_refs:
                    try:
                        ref_dict[group[key_columns.index('attribute_name')]].add('[' + group[-1] + '](' + '-'.join(
                            supported_name[
                                measure]
                                .lower().split(' ')) + ')')
                    except KeyError:
                        ref_dict[group[key_columns.index('attribute_name')]] = set()
                        ref_dict[group[key_columns.index('attribute_name')]].add('[' + group[-1] + '](' + '-'.join(
                            supported_name[
                                measure].lower().split(
                                ' ')) + ')')
            cellref = ''
            align_rows = True if attr_order else False
            refs_order = attr_order if attr_order else ref_dict.keys()
            newline = '##br##'
            idented = '&emsp;&emsp;&emsp;'
            for key in refs_order:
                cellref += '**{attribute_name}:**'.format(attribute_name=key) + newline
                cellref += idented + list(ref_dict[key])[0] + ' ##br##'
                if align_rows:
                    cellref += ''.join([newline] * (len(false_dict[key]) - 1))  # this is the number of lines to skip in cell
                else:
                    cellref += newline
            raw['Reference Groups Selected'].append(cellref)

    highlevel_table = '\n\n'
    if len(raw['Fairness Criteria']) > 0:
        landf = pd.DataFrame(raw, columns=['Fairness Criteria', 'Desired Outcome', 'Reference Groups Selected',
                                           'Unfairly Affected Groups'])
        # keep the same order!!
        # krrp
        highlevel_table = tabulate(landf[['Fairness Criteria', 'Desired Outcome', 'Reference Groups Selected',
                                          'Unfairly Affected Groups']], headers='keys',
                                   tablefmt='pipe', showindex='never', numalign="left")

    return highlevel_table



def audit_summary(configs, group_value_df):
    supported_name2 = {'Statistical Parity': 'Equal Parity',
                       'Impact Parity': 'Proportional Parity',
                       'FPR Parity': 'False Positive Rate Parity',
                       'FDR Parity': 'False Discovery Rate Parity',
                       'FNR Parity': 'False Negative Rate Parity',
                       'FOR Parity': 'False Omission Rate Parity'}

    supported_name = {'Statistical Parity': '**Equal Parity** - Ensure all protected groups are have equal representation in the selected set.',
                      'Impact Parity': '**Proportional Parity** - Ensure all protected groups are selected proportional to their '
                                       'percentage of the population.',
                      'FPR Parity': '**False Positive Rate Parity** - Ensure all protected groups have the same false positive '
                                    'rates as the reference group).',
                      'FDR Parity': '**False Discovery Rate Parity** - Ensure all protected groups have equally proportional false '
                                    'positives within the selected set (compared to the reference group).',
                      'FNR Parity': '**False Negative Rate Parity** - Ensure all protected groups have the same false negative rates ('
                                    'as the reference group).',
                      'FOR Parity': '**False Omission Rate Parity** - Ensure all protected groups have equally proportional false '
                                    'negatives within the non-selected set (compared to the reference group).'}

    raw = {
        'column1': [],
        'column2': [],
        'column3': []
    }
    supported_order = ['Statistical Parity', 'Impact Parity', 'FPR Parity', 'FDR Parity', 'FNR Parity', 'FOR Parity']
    measures_results_dict = {}
    for measure in supported_order:
        if measure in configs.fair_measures_requested:
            raw['column1'].append(supported_name[measure])
            false_df = group_value_df.loc[group_value_df[measure] == False]
            if false_df.empty:
                link = ' [Details](#' + '-'.join(supported_name2[measure].lower().split(' ')) + \
                       '-span-green-initpassedspan-green-end)'
                raw['column2'].append('++span-green-init++**Passed**++span-green-end++')
                raw['column3'].append(link)

                measures_results_dict[measure] = '++span-green-init++Passed++span-green-end++'
            else:
                link = ' [Details](#' + '-'.join(supported_name2[measure].lower().split(' ')) + \
                       '-span-red-initfailedspan-red-end)'
                raw['column2'].append('++span-red-init++**Failed**++span-red-end++')
                raw['column3'].append(link)
                measures_results_dict[measure] = '++span-red-init++Failed++span-red-end++'
    df = pd.DataFrame(raw, columns=['column1', 'column2', 'column3'])
    summ_table = tabulate(df[['column1', 'column2', 'column3']], headers='keys',
                          tablefmt='pipe', showindex='never', numalign="left")
    return summ_table, measures_results_dict


def audit_description(configs, group_value_df):

    supported_name = {'Statistical Parity': '**Equal Parity** - Ensure all protected groups are have equal representation in the selected set.',
                      'Impact Parity': '**Proportional Parity** - Ensure all protected groups are selected proportional to their '
                                       'percentage of the population.',
                      'FPR Parity': '**False Positive Rate Parity** - Ensure all protected groups have the same false positive '
                                    'rates as the reference group).',
                      'FDR Parity': '**False Discovery Rate Parity** - Ensure all protected groups have equally proportional false '
                                    'positives within the selected set (compared to the reference group).',
                      'FNR Parity': '**False Negative Rate Parity** - Ensure all protected groups have the same false negative rates ('
                                    'as the reference group).',
                      'FOR Parity': '**False Omission Rate Parity** - Ensure all protected groups have equally proportional false '
                                    'negatives within the non-selected set (compared to the reference group).'}

    supported_order = ['Statistical Parity', 'Impact Parity', 'FPR Parity', 'FDR Parity', 'FNR Parity', 'FOR Parity']

    ref_group = {'predefined': 'Custom group - The reference groups you selected for each attribute will be used  to '
                               'calculate relative disparities in this audit.',
                 'majority': 'Majority group - The largest groups on each attribute will be used as baseline to calculate '
                             'relative '
                             'disparities in this audit.',
                 'min_metric': 'Automatically select, for each bias metric, the group on each attribute that has the '
                               'lower '
                               'value, to be used as baseline to calculate relative disparities in this audit.'
                 }

    raw = {
        'column1': [],
        'column2': []
    }

    raw['column1'].append('**Audit Date:**')
    raw['column2'].append(datetime.date.today().strftime('%d %b %Y'))
    raw['column1'].append('**Data Audited:**')
    raw['column2'].append('{:.0f} rows'.format(group_value_df['total_entities'].values[0]))
    raw['column1'].append('**Attributes Audited:**')
    raw['column2'].append(', '.join(group_value_df['attribute_name'].unique()))
    raw['column1'].append('**Audit Goal(s):**')
    measures = [supported_name[m] for m in supported_order if m in configs.fair_measures_requested]
    raw['column2'].append('\n'.join(measures) + '\n')
    raw['column1'].append('**Reference Groups:**')

    raw['column2'].append(ref_group[configs.ref_groups_method])
    raw['column1'].append('**Fairness Threshold:**')
    thresh = '{:.0f}%. If disparity for a group is within {' \
             ':.0f}% and ' \
             '{' \
             ':.0f}% of ' \
             ' the value of the reference group on a group metric (e.g. False ' \
             'Positive Rate), this audit will pass. ' \
             ''.format(
        float(configs.fairness_threshold) * 100, float(configs.fairness_threshold) * 100,
        float(1.0 / configs.fairness_threshold) * 100)
    raw['column2'].append(thresh)

    df = pd.DataFrame(raw, columns=['column1', 'column2'])
    desc_table = tabulate(df[['column1', 'column2']], headers='keys',
                          tablefmt='pipe', showindex='never', numalign="left")

    return desc_table

def audit_report_markdown(configs, group_value_df, fairness_measures_depend, overall_fairness, model_id=1):
    manylines = '\n\n&nbsp;\n\n&nbsp;\n\n'
    oneline = ' \n\n&nbsp;\n\n'
    mkdown_highlevel = '# The Bias Report'
    mkdown_highlevel += oneline
    mkdown_highlevel += audit_description(configs, group_value_df) + oneline + '----' + oneline

    # mkdown_highlevel += get_sentence_highlevel(overall_fairness) + oneline
    # mkdown_highlevel += get_highlevel_table(group_value_df, configs.fair_measures_requested) + oneline + '----' + oneline

    mkdown_highlevel += '### Audit Results:\n\n'
    mkdown_highlevel += '1. [Summary](#audit-results-summary)\n\n'
    mkdown_highlevel += '2. [Details by Fairness Measures](#audit-results-details-by-fairness-measures)\n\n'
    mkdown_highlevel += '3. [Details by Protected Attributes](#audit-results-details-by-protected-attributes)\n\n'
    mkdown_highlevel += '4. [Bias Metrics Values](#audit-results-bias-metrics-values)\n\n'
    mkdown_highlevel += '5. [Base Metrics Calculated for Each Group](#audit-results-group-metrics-values)\n\n' + oneline + '----' + \
                        oneline

    mkdown_highlevel += '### Audit Results: Summary' + '\n\n'
    summ_table, measures_results_dict = audit_summary(configs, group_value_df)
    mkdown_highlevel += summ_table + oneline + '----' + oneline

    mkdown_highlevel += '### Audit Results: Details by Fairness Measures' + oneline

    if 'Statistical Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n#### Equal Parity: ' + measures_results_dict['Statistical Parity'] + '\n\n'
        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have equal parity is every group is equally '
                              'represented in the selected set. For example, if race (with possible values of white, black, other) '
                              'has equal parity, it implies that all three races are equally represented (33% each)'
                              'in the selected/intervention set.']
        raw['When does it matter?'] = ['##border##If your desired ' \
                                       'outcome is to intervene equally on people ' \
                                                         'from all races, then you care about this criteria.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'Statistical Parity',
                                                                              fairness_measures_depend)
        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(dft[['What is it?', 'When does it matter?', 'Which groups failed the audit:']],
                                     headers='keys',
                                     tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        # mkdown_highlevel += '**The Bias Report has found that the following groups do not have Equal Parity:**\n\n'
        #mkdown_highlevel += get_statpar_text(group_value_df, fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline + '----' + oneline

    if 'Impact Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n#### Proportional Parity: ' + measures_results_dict['Impact Parity'] + '\n\n'
        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have proportional parity if every group is ' \
                              'represented proportionally to their share of the population. For example, if race ' \
                              'with possible values of white, black, other being 50%, 30%, 20% of the population respectively) has ' \
                              'proportional parity, it implies that all three races are represented in the same proportions ' \
                              '(50%, 30%, 20%) in the selected set.']
        raw['When does it matter?'] = ['##border##If your desired outcome is to intervene ' \
                                       'proportionally on people from all races, then you care about this criteria.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'Impact Parity',
                                                                              fairness_measures_depend)

        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(dft[['What is it?', 'When does it matter?', 'Which groups failed the audit:']],
                                     headers='keys',
                                     tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        #mkdown_highlevel += get_impact_text(group_value_df, fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline + '----' + oneline

    if 'FPR Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n#### False Positive Rate Parity: ' + measures_results_dict['FPR Parity'] + '\n\n'
        # mkdown_highlevel += 'False Positive Rate Parity is concerned with Type I errors (False Positives). In cases ' \
        #                     'of punitive ' \
        #                     'interventions on the selected set' \
        #                     ' it is important to not have disparate Type I errors across groups. Aequitas audits both False ' \
        #                     'Positive Rate (FP/Negative Labels of each group) and False Discovery Rates (FP/Selected ' \
        #                     'Set Size).\n\n'

        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have False Positive '
                              'parity if '
                              'every group ' \
                              'has the same False Positive Error Rate. For example, if race has false positive parity, ' \
                              'it implies that all three races have the same False Positive Error Rate.']
        raw['When does it matter?'] = [
            '##border##If your desired outcome is to make false positive '
                                                                       'errors ' \
                                                                       'equally on people from all races, then you care about this criteria. This is important in cases where your intervention is ' \
                                                                       'punitive ' \
                                                                       'and has a risk of adverse outcomes for individuals. Using this criteria allows you to make sure that ' \
                                                                       'you are not making false positive mistakes about any single group disproportionately.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'FPR Parity',
                                                                              fairness_measures_depend)

        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(
            dft[['What is it?', 'When does it matter?', 'Which groups failed the audit:']],
            headers='keys',
            tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline
    mkdown_highlevel += oneline + '----' + oneline

    if 'FDR Parity' in group_value_df.columns:
        mkdown_highlevel += oneline + '\n\n#### False Discovery Rate Parity: ' + measures_results_dict['FDR Parity'] + '\n\n'
        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have False Discovery Rate parity if every group ' \
                              'has the same False Discovery Error Rate. For example, if race has false discovery parity, ' \
                              'it implies that all three races have the same False Discvery Error Rate.']
        raw['When does it matter?'] = [
            '##border##If your desired outcome is to make false positive '
                                                                        'errors ' \
                                                                        'equally on people from all races, then you care about this criteria. This is important in cases where your intervention is ' \
                                                                        'punitive and can hurt individuals and where you are selecting a very small group for interventions.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'FDR Parity',
                                                                              fairness_measures_depend)

        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(
            dft[['What is it?', 'When does it matter?', 'Which groups failed the audit:']],
            headers='keys',
            tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline
    mkdown_highlevel += oneline + '----' + oneline

    if 'FNR Parity' in group_value_df.columns:
        # mkdown_highlevel += 'False Negative Parity is concerned with Type II errors (False Negatives). In cases ' \
        #                     'of assistive or preventive ' \
        #                     'interventions on the selected set' \
        #                     ' it is important to not have disparate Type II errors across groups. Aequitas audits both False ' \
        #                     'Negative Rate (FN/Positive Labels of each group) and False Omission Rates (FN/Not Selected ' \
        #                     'Set Size).\n\n'

        mkdown_highlevel += oneline + '\n\n#### False Negative Rate Parity: ' + measures_results_dict['FNR Parity'] + '\n\n'
        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have False Negative parity if every group ' \
                              'has the same False Negative Error Rate. For example, if race has false negative parity, it implies that all three ' \
                              'races ' \
                              'have the same False Negative Error Rate.']
        raw['When does it matter?'] = [
            '##border##If your desired outcome is to make false negative errors equally on ' \
            'people from all races, then you care about this criteria. This is important in cases where your intervention is ' \
            'assistive (providing helpful social services for example) and missing an individual could lead to adverse outcomes for them. Using this criteria allows you to make ' \
            'sure ' \
            'that you’re not missing people from certain groups '
            'disproportionately.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'FNR Parity',
                                                                              fairness_measures_depend)
        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(dft[['What is it?', 'When does it matter?',
                                          'Which groups failed the audit:']],
                                     headers='keys',
                                     tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline
    mkdown_highlevel += oneline + '----' + oneline

    if 'FOR Parity' in group_value_df.columns:
        mkdown_highlevel += oneline + '\n\n#### False Omission Rate Parity: ' + measures_results_dict['FOR Parity'] + '\n\n'
        raw = {}
        raw['What is it?'] = ['##border##This criteria considers an attribute to have False Omission Rate parity if every group ' \
                              'has the same False Omission Error Rate. For example, if race has false omission parity, it implies that all three ' \
                              'races ' \
                              'have the same False Omission Error Rate.']
        raw['When does it matter?'] = [
            '##border##If your desired outcome is to make false negative '
                                                                       'errors ' \
                                                                       'equally ' \
                                                                       'on people from all races, then you care about this criteria. This is important in cases where your intervention is ' \
                                                                       'assistive (providing help social services for example) and missing an individual could lead to adverse outcomes for them '\
                                                                       ', and where you are selecting a very small group for interventions. Using this criteria allows you to make ' \
                                                                       'sure ' \
                                                                       'that you’re not missing people from certain groups '
                                                                       'disproportionately.']
        raw['Which groups failed the audit:'] = '##border##' + get_false_text(group_value_df, 'FOR Parity',
                                                                              fairness_measures_depend)

        dft = pd.DataFrame(raw)
        mkdown_highlevel += tabulate(dft[['What is it?', 'When does it matter?',
                                          'Which groups failed the audit:']],
                                     headers='keys',
                                     tablefmt='pipe', showindex='never', numalign="left") + \
                            oneline

        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline
    # mkdown_highlevel += oneline + '----' + oneline

    mkdown_parity = '\n\n### Audit Results: Details by Protected Attributes' + oneline
    # do we want to show this?
    # mkdown_parity += get_highlevel_report(group_attribute_df) + '\n\n'

    mkdown_disparities = '\n\n### Audit Results: Bias Metrics Values' + oneline
    mkdown_group = '\n\n### Audit Results: Group Metrics Values' + oneline
    # setup the group_value_df (colors and stuff)
    group_value_df['group_size_pct'] = group_value_df['group_size'].divide(group_value_df['total_entities'])
    group_value_df = setup_group_value_df(group_value_df, configs.fair_measures_requested,
                                          fairness_measures_depend)
    for attr in configs.attr_cols:
        mkdown_parity += '\n\n#### ' + attr + oneline
        mkdown_disparities += '\n\n#### ' + attr + oneline
        mkdown_group += '\n\n#### ' + attr + oneline
        mkdown_parity += get_parity_group_report(group_value_df, attr, configs.fair_measures_requested, fairness_measures_depend)
        mkdown_disparities += get_disparities_group_report(group_value_df, attr, configs.fair_measures_requested,
                                                           fairness_measures_depend)
        mkdown_group += get_group_group_report(group_value_df, attr, configs.fair_measures_requested,
                                               fairness_measures_depend)
        mkdown_parity += '\n\n[Go to Top](#)' + manylines
        mkdown_disparities += '\n\n[Go to Previous]' + '(#' + '-'.join(attr.lower().split(' ')) + \
                              ')' + '\n\n' + '[Go to Top](#)' + manylines
        mkdown_group += '\n\n[Go to Previous]' + '(#' + '-'.join(attr.lower().split(' ')) + \
                        '-2)' + '\n\n' + '[Go to Top](#)' + manylines

    report = mkdown_highlevel + '----' + mkdown_parity + '----' + mkdown_disparities + '----' + mkdown_group + '----'
    report_html = markdown(report, extras=['tables', 'header-ids'])
    # coloring True/False results
    report_html = report_html.replace('nan', 'Undefined')
    report_html = report_html.replace('>False<', ' style="color:red"><b>Failed</b><')
    report_html = report_html.replace('>True<', ' style="color:green"><b>Passed</b><')

    report_html = report_html.replace('>##red##', ' style="color:red">')
    report_html = report_html.replace('>##green##', ' style="color:green">')

    report_html = report_html.replace('##br##', '<br>')

    report_html = report_html.replace('++span-red-init++', '<span style="color:red"><b>')
    report_html = report_html.replace('++span-red-end++', '</b> </span> ')
    report_html = report_html.replace('++span-green-init++', '<span style="color:green"><b>')
    report_html = report_html.replace('++span-green-end++', '</b> </span> ')

    # report_html = report_html.replace(' failed ', '<span style="color:red"><b> failed </b></span>')
    # report_html = report_html.replace(' passed ', '<span style="color:green"><b> passed </b></span>')

    report_html = report_html.replace('<table>', '<table class="table table-striped" padding=5 >')
    report_html = report_html.replace('<h1 id="the-bias-report">', '<h1 id="the-bias-report" align="center">')
    # report_html = report_html.replace('< thead >\n < tr >', '< thead >\n < tr class="table-info" >')
    #     audit_desc = """<table class="table table-striped" padding=5 >
    # <thead>
    # <tr>
    #   <th align="left">column1</th>
    #   <th align="left">column2</th>
    #   <th align="left">column3</th>
    # </tr>
    # </thead>"""
    #     new_audit_desc = '<table>\n'
    #     report_html = report_html.replace(audit_desc, new_audit_desc)


    ##TOOLTIPS
    # report_html = report_html.replace('Statistical Parity', ' <a href="#" data-toggle="tooltip" title="Hooray!">Equal
    # Parity</a>')

    ##summary table we need a different replace
    report_html = report_html.replace('<td align="left" style="color:red">', '<td align="left" style="color:red; '
                                                                             'padding:10px;">')
    report_html = report_html.replace('<td align="left" style="color:green">', '<td align="left" style="color:green; '
                                                                               'padding:10px;">')
    report_html = report_html.replace('<td align="left">', '<td align="left" style=" padding:5px;">')

    report_html = report_html.replace(';">##border##', '; border: 15px solid white;">')

    ## widths tables
    width1_default = '<th align="left">What is it?</th>'
    width2_default = '<th align="left">Which groups failed the audit:</th>'
    width1_new = '<th align="left" width="35%" >What is it?</th>'
    width3_new = '<th align="left" width="30%" >Which groups failed the audit:</th>'
    report_html = report_html.replace(width1_default, width1_new)
    report_html = report_html.replace(width2_default, width3_new)

    report_html = report_html.replace('##tooltip-start-title##', '<a href="javascript:;" data-toggle="tooltip" title="')
    report_html = report_html.replace('##tooltip-end-title##', ' ">')
    report_html = report_html.replace('##tooltip-end-anchor##', '</a>')

    report_html = report_html.replace('>Statistical Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                              'title="Fail/Pass test if the Predicted Positive Rate '
                                                              'Disparity of each group is within the range allowed by the fairness '
                                                              'threshold selected.">Equal Parity</a><')



    report_html = report_html.replace('>Impact Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                         'title="Fail/Pass test if the Predicted Positive Group Rate '
                                                         'Disparity of each group is within the range allowed by the fairness threshold selected.">Proportional Parity</a><')

    report_html = report_html.replace('>FPR Parity<', ' ><a href="javascript:;" data-toggle="tooltip" title="Fail/Pass test '
                                                      'if the False Positive Rate Disparity of each group is within the range '
                                                      'allowed by the fairness threshold selected.">False '
                                                      'Positive '
                                                      'Rate Parity</a><')
    report_html = report_html.replace('>FDR Parity<', '> <a href="javascript:;" data-toggle="tooltip" title="Fail/Pass test '
                                                      'if the False Discovery Rate Disparity of each group is within the range '
                                                      'allowed by the fairness threshold selected.">False Discovery '
                                                      'Rate Parity</a><')
    report_html = report_html.replace('>FNR Parity<', ' ><a href="javascript:;" data-toggle="tooltip" title="Fail/Pass test '
                                                      'if the False Negative Rate Disparity of each group is within the range '
                                                      'allowed by the fairness threshold selected.">False Negative '
                                                      'Rate Parity</a><')
    report_html = report_html.replace('>FOR Parity<', ' ><a href="javascript:;" data-toggle="tooltip" title="Fail/Pass test '
                                                      'if the False Omission Rate Disparity of each group is within the range '
                                                      'allowed by the fairness threshold selected.">False '
                                                      'Omission '
                                                      'Rate Parity</a><')

    report_html = report_html.replace('>Equal Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                        'title="Fail/Pass test if the Predicted Positive Rate '
                                                        'Disparity of each group is within the range allowed by the fairness '
                                                        'threshold selected.">Equal Parity</a><')


    report_html = report_html.replace('>Proportional Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                               'title="Fail/Pass test if the Predicted Positive Group Rate '
                                                               'Disparity of each group is within the range allowed by the fairness threshold selected.">Proportional Parity</a><')

    report_html = report_html.replace('>False Positive Rate Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                                      'title="Fail/Pass test '
                                                                      'if the False Positive Rate Disparity of each group is within the range '
                                                                      'allowed by the fairness threshold selected.">False '
                                                                      'Positive '
                                                                      'Rate Parity</a><')
    report_html = report_html.replace('>False Discovery Rate Parity<', '> <a href="javascript:;" data-toggle="tooltip" '
                                                                       'title="Fail/Pass test '
                                                                       'if the False Discovery Rate Disparity of each group is within the range '
                                                                       'allowed by the fairness threshold selected.">False Discovery '
                                                                       'Rate Parity</a><')
    report_html = report_html.replace('>False Negative Rate Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                                      'title="Fail/Pass test '
                                                                      'if the False Negative Rate Disparity of each group is within the range '
                                                                      'allowed by the fairness threshold selected.">False Negative '
                                                                      'Rate Parity</a><')
    report_html = report_html.replace('>False Omission Rate Parity<', ' ><a href="javascript:;" data-toggle="tooltip" '
                                                                      'title="Fail/Pass test '
                                                                      'if the False Omission Rate Disparity of each group is within the range '
                                                                      'allowed by the fairness threshold selected.">False '
                                                                      'Omission '
                                                                      'Rate Parity</a><')

    report_html = report_html.replace('>PPREV Disparity<',
                                      ' ><a href="javascript:;" data-toggle="tooltip" title="The Predicted Positive Group Ratio of each group divided by the same metric value of the reference group. '
                                      '.">Predicted Positive '
                                      'Group Rate Disparity'
                                      '</a><')

    report_html = report_html.replace('>PPR Disparity<', ' ><a href="javascript:;" data-toggle="tooltip" title="The Predicted '
                                                         'Positive Rate of each group divided by the same metric value of the reference group.">Predicted '
                                                         'Positive '
                                                         'Rate Disparity</a><')

    report_html = report_html.replace('>FPR Disparity<', ' ><a href="javascript:;" data-toggle="tooltip" title="The False '
                                                         'Positive Rate of each group divided by the same metric value of the '
                                                         'reference group.">False '
                                                         'Positive '
                                                         'Rate Disparity</a><')
    report_html = report_html.replace('>FDR Disparity<', '> <a href="javascript:;" data-toggle="tooltip" title="The False '
                                                         'Discovery Rate of each group divided by the same metric value of the '
                                                         'reference group.">False Discovery '
                                                         'Rate Disparity</a><')
    report_html = report_html.replace('>FNR Disparity<', ' ><a href="javascript:;" data-toggle="tooltip" title="The False '
                                                         'Negative Rate of each group divided by the same metric value of the '
                                                         'reference group.">False Negative '
                                                         'Rate Disparity</a><')
    report_html = report_html.replace('>FOR Disparity<', ' ><a href="javascript:;" data-toggle="tooltip" title="The False '
                                                         'Omission Rate of each group divided by the same '
                                                         'metric value of the reference group.">False '
                                                         'Omission '
                                                         'Rate Disparity</a><')

    report_html = report_html.replace('>PPREV<',
                                      ' ><a href="javascript:;" data-toggle="tooltip" title="Number of elements of the '
                                      'group selected divided by the size of the group.">Predicted Positive '
                                      'Group Rate'
                                      '</a><')

    report_html = report_html.replace('>PPR<', ' ><a href="javascript:;" data-toggle="tooltip" title="Number of elements of the '
                                               'group selected divided by the total size of the interventation set.">Predicted '
                                               'Positive '
                                               'Rate</a><')

    report_html = report_html.replace('>FPR<',
                                      ' ><a href="javascript:;" data-toggle="tooltip" title="Number of false positives of '
                                      'the group divided by number of labeled negatives within the group.">False '
                                      'Positive '
                                      'Rate</a><')
    report_html = report_html.replace('>FDR<',
                                      '> <a href="javascript:;" data-toggle="tooltip" title="Number of false positives of '
                                      'the group divided by the size of the intervention set of the group (predicted '
                                      'positives within group).">False Discovery '
                                      'Rate</a><')
    report_html = report_html.replace('>FNR<',
                                      ' ><a href="javascript:;" data-toggle="tooltip" title="Number of false negative of '
                                      'the group divided by number of labeled positives in the group">False Negative '
                                      'Rate</a><')
    report_html = report_html.replace('>FOR<',
                                      ' ><a href="javascript:;" data-toggle="tooltip" title="Number of false negatives of '
                                      'the group divided by the size of the non-selected set of the group (predicted '
                                      'negatives within group).">False '
                                      'Omission '
                                      'Rate</a><')

    audit_desc = """<table class="table table-striped" padding=5 >
<thead>
<tr>
  <th align="left">column1</th>
  <th align="left">column2</th>
</tr>
</thead>"""
    new_audit_desc = '<table>\n'
    report_html = report_html.replace(audit_desc, new_audit_desc)

    audit_desc = """<table class="table table-striped" padding=5 >
<thead>
<tr>
  <th align="left">column1</th>
  <th align="left">column2</th>
  <th align="left">column3</th>
</tr>
</thead>"""
    new_audit_desc = '<table>\n'
    report_html = report_html.replace(audit_desc, new_audit_desc)
    # change color of headers
    report_html = report_html.replace('<h1', '<h1 style="color:#b37d4e;"')
    report_html = report_html.replace('<h3', '<h3 style="color:#b37d4e;"')
    report_html = report_html.replace('<h3', '<h3 style="color:#b37d4e;"')
    report_html = report_html.replace('<h4', '<h4 style="color:#286da8;"')
    report_html = report_html.replace('<strong ><a', '<a')
    report_html = report_html.replace('</a></strong>', '</a>')
    report_html = report_html.replace('<strong> <a', '<a')

    return report_html
