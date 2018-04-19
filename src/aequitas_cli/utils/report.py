import logging

import pandas as pd
from markdown2 import markdown
from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True

logging.getLogger(__name__)

# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.


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
                ppr_text = '{:.2f}% of the group is selected, compared to {:.2f} % of the ' \
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
        map[col] = colstr  # + ' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        # to be able to click on true/false and redirect to the next section
        if col != 'attribute_name':
            for attr in attr_list:
                group_attribute_df.loc[group_attribute_df['attribute_name'] == attr, col] = '[' + group_attribute_df[col][
                    group_attribute_df['attribute_name'] == attr] + ']' + '(#' + '-'.join(attr.lower().split(' ')) + ')'
    for attr in attr_list:
        group_attribute_df = group_attribute_df.replace(attr, '[' + attr + ']' + '(#' + '-'.join(attr.lower().split(' ')) + ')')
    group_attribute_df = group_attribute_df.rename(index=str, columns=map)
    highlevel_report = tabulate(group_attribute_df, headers='keys', tablefmt='pipe', showindex='never')
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
                            tablefmt='pipe', showindex='never')
    return parity_group


def setup_group_value_df(group_value_df, fairness_measures, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    metrics = {}
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            metrics[disp] = par
            metrics[disp.replace('_disparity', '')] = par
    aux_df = group_value_df[['attribute_name', 'attribute_value'] + list(metrics.keys())]
    print('\n\n', aux_df)

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
                                 tablefmt='pipe', showindex='never')

    return disparities_group


def get_group_group_report(group_value_df, attribute, fairness_measures, fairness_measures_depend):
    def_cols = ['attribute_value']
    for par, disp in fairness_measures_depend.items():
        if par in fairness_measures:
            def_cols.append(disp.replace('_disparity', ''))
    aux_df = group_value_df.loc[group_value_df['attribute_name'] == attribute]
    aux_df = aux_df[def_cols]
    aux_df = aux_df.round(2)
    # fixing the same order of columns every time!
    cols_order = ['attribute_value', 'ppr', 'pprev', 'fdr', 'fpr', 'for', 'fnr']
    new_order = [col for col in cols_order if col in aux_df.columns]
    aux_df = aux_df[new_order]
    map = {}
    for col in aux_df.columns:
        if col == 'attribute_value':
            colstr = 'Attribute Value'
        else:
            colstr = col.upper()
        map[col] = colstr  #+ ' &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
    aux_df = aux_df.rename(index=str, columns=map)
    group_group = tabulate(aux_df,
                           headers='keys',
                           tablefmt='pipe', showindex='never')
    return group_group


def get_sentence_highlevel(fair_results):
    sent = '#### The Bias Report evaluates the current model as'
    if fair_results['Overall Fairness'] is True:
        is_fair = ' fair'
    else:
        is_fair = ' unfair'  # ' unfair to the following groups: '
    sent += is_fair + ' using the following fairness criteria:\n\n'
    return sent


def get_false_text(group_value_df, fairness_metric, fairness_measures_depend):
    names = {

        'fpr': 'false positive rate',
        'fnr': 'false negative rate',
        'fdr': 'false discovery rate',
        'for': 'false omission rate',
        'ppr': 'predicted positive ratio',
        'pprev': 'predicted prevalence (base rate)'
    }
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]
        sentence = 'The {group_metric_name} for \"{attribute_name} = {attribute_value}\" is {bias_metric_value}% ' \
                   'of the {group_metric_name} of the reference group \"{attribute_name} = {ref_group_value}\",' \
                   ' corresponding to a difference of {group_metric_value} vs {ref_group_metric_value}.' \
            .format(attribute_name=row['attribute_name'],
                    attribute_value=row['attribute_value'],
                    group_metric_name=names[group_metric],
                    bias_metric_value='%.2f' % (float(row[bias_metric]) * 100),
                    ref_group_value=row[ref_group_col],
                    group_metric_value=row[group_metric],
                    ref_group_metric_value=ref_group_row[group_metric].values[0])
        text_detail += sentence + '\n\n'
    if false_df.empty:
        text_detail += 'Based on the fairness threshold used, there is no disparate values in {group_metric_name} between ' \
                       'the each group and the respective reference group.\n\n'.format(group_metric_name=names[
            group_metric])
    return text_detail


def get_statpar_text(group_value_df, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    fairness_metric = 'Statistical Parity'
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]

        sentence = 'The number of elements of the group  \"{attribute_name} = {attribute_value}\" selected as positive ' \
                   'represents {group_metric_value}% of the ' \
                   'total number of positives across groups, while ' \
                   'in the case of the reference group \"{attribute_name} = {ref_group_value}\" the number of positives ' \
                   'correspond to {ref_group_metric_value}% of the total positives.' \
                   ''.format(attribute_name=row['attribute_name'],
                             attribute_value=row['attribute_value'],
                             group_metric_value='%.2f' % (float(row[group_metric]) * 100),
                             ref_group_metric_value='%.2f' % (float(ref_group_row[group_metric].values[0]) * 100),
                             ref_group_value=row[ref_group_col])
        text_detail += sentence + '\n\n'
    if false_df.empty:
        text_detail += 'Based on the fairness threshold used, the number of selected positives is similar across ' \
                       'different ' \
                       'groups.\n\n'
    return text_detail


def get_impact_text(group_value_df, fairness_measures_depend):
    group_value_df = group_value_df.round(2)
    group_value_df = group_value_df.applymap(str)
    fairness_metric = 'Impact Parity'
    false_df = group_value_df.loc[group_value_df[fairness_metric] == 'False']
    bias_metric = fairness_measures_depend[fairness_metric]
    group_metric = bias_metric.replace('_disparity', '')
    ref_group_col = group_metric + '_ref_group_value'
    text_detail = ''
    for index, row in false_df.iterrows():
        ref_group_row = group_value_df.loc[(group_value_df['attribute_name'] == row['attribute_name']) &
                                           (group_value_df['attribute_value'] == row[ref_group_col])]

        sentence = '{group_metric_value}% of the group \"{attribute_name}:{attribute_value}\" is considered positive,' \
                   ' in comparison to {ref_group_metric_value}% of positives within the reference group \"{attribute_name} = {' \
                   'ref_group_value}\"' \
                   ''.format(
            group_metric_value='%.2f' % (float(row[group_metric]) * 100),
            attribute_name=row['attribute_name'],
            attribute_value=row['attribute_value'],
            ref_group_metric_value='%.2f' % (float(ref_group_row[group_metric].values[0]) * 100),
            ref_group_value=row[ref_group_col])

        text_detail += sentence + '\n\n'
    if false_df.empty:
        text_detail += 'Based on the fairness threshold used, the percentage of selected elements from ' \
                       'each group is not disparate to the percentage of selected elements of the respective reference group.\n\n'
    return text_detail


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
            for key in ref_dict.keys():
                cellref += '**{attribute_name}:** ##br##&emsp;&emsp;&emsp;'.format(attribute_name=key)
                cellref += '##br##&emsp;&emsp;&emsp;'.join(ref_dict[key]) + ' ##br##'
            raw['Reference Groups Selected'].append(cellref)

            for index, row in false_df.iterrows():
                try:

                    false_dict[row['attribute_name']].add('[' + row['attribute_value'] + ']' + \
                                                          supported_name[measure][supported_name[measure].find('('):])
                except KeyError:
                    false_dict[row['attribute_name']] = set()
                    false_dict[row['attribute_name']].add('[' + row['attribute_value'] + ']' + \
                                                          supported_name[measure][supported_name[measure].find('('):])
            if len(false_dict) > 0:
                cell = ''
                for key in false_dict.keys():
                    cell += '**{attribute_name}:** ##br##&emsp;&emsp;&emsp;'.format(attribute_name=key)
                    cell += '##br##&emsp;&emsp;&emsp;'.join(false_dict[key]) + ' ##br##'

                raw['Unfairly Affected Groups'].append(cell)
            else:
                if group_value_df[measure].isnull().all():
                    raw['Unfairly Affected Groups'].append('Undefined')
                else:
                    raw['Unfairly Affected Groups'].append('No Unfair Groups Found')

    highlevel_table = '\n\n'
    if len(raw['Fairness Criteria']) > 0:
        landf = pd.DataFrame(raw, columns=['Fairness Criteria', 'Desired Outcome', 'Reference Groups Selected',
                                           'Unfairly Affected Groups'])
        # keep the same order!!
        # krrp
        highlevel_table = tabulate(landf[['Fairness Criteria', 'Desired Outcome', 'Reference Groups Selected',
                                          'Unfairly Affected Groups']], headers='keys',
                                   tablefmt='pipe', showindex='never')
        print(highlevel_table)
    return highlevel_table


def audit_report_markdown(configs, group_value_df, fairness_measures_depend, overall_fairness, model_id=1):
    manylines = '\n\n&nbsp;\n\n&nbsp;\n\n'
    oneline = ' \n\n&nbsp;\n\n'
    mkdown_highlevel = '# The Bias Report' + manylines + oneline
    mkdown_highlevel += '#### Fairness Threshold: {:.0f}%'.format(float(configs.fairness_threshold) * 100) + oneline
    mkdown_highlevel += get_sentence_highlevel(overall_fairness) + oneline
    mkdown_highlevel += get_highlevel_table(group_value_df, configs.fair_measures_requested) + '.' + oneline + '----'

    mkdown_highlevel += oneline + '### Table of Contents:\n\n'
    mkdown_highlevel += '1. [Fairness Overview](#fairness-criteria-assessments)\n\n'
    mkdown_highlevel += '2. [Fairness Criteria Assessments](#fairness-criteria-assessments)\n\n'
    mkdown_highlevel += '3. [Some Numbers: Bias Metrics](#some-numbers:-bias-metrics)\n\n'
    mkdown_highlevel += '4. [More Numbers: Group Metrics](#more-numbers:-group-metrics)\n\n' + oneline + '----' + manylines

    mkdown_highlevel += '## Fairness Overview' + oneline
    if 'Statistical Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n### Equal Parity\n\n' + oneline
        mkdown_highlevel += """**What is it?** This criteria considers an attribute to have equal parity is every group is equally 
                            represented in the selected set. For example, if race (with possible values of white, black, other) 
                            has equal parity, it implies that all three races are equally represented (33% each) 
                            in the selected/intervention set.\n\n**When should I care about Equal Parity?** If your desired outcome is to intervene equally on people 
                            from all races, then you care about this criteria.\n\n""" + oneline
        mkdown_highlevel += '**The Bias Report has found that the following groups do not have Equal Parity:**\n\n'
        mkdown_highlevel += get_statpar_text(group_value_df, fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline

    if 'Impact Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n### Proportional Parity\n\n'
        mkdown_highlevel += """**What is it?** This criteria considers an attribute to have proportional parity if every group is 
                            represented proportionally to their share of the population. For example, if race 
                            (with possible values of white, black, other being 50%, 30%, 20% of the population respectively) has 
                            proportional parity, it implies that all three races are represented in the same proportions 
                            (50%, 30%, 20%) in the selected set.\n\n**When should I care about Proportional Parity?** If your desired outcome is to intervene 
                            proportionally on people from all races, then you care about this criteria.\n\n""" + oneline
        mkdown_highlevel += '**The Bias Report has found that the following groups do not have Proportional Parity:**\n\n'
        mkdown_highlevel += get_impact_text(group_value_df, fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline

    if 'TypeI Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n### False Positive Parity\n\n'
        mkdown_highlevel += """**What is it?** This criteria considers an attribute to have False Positive parity if every group 
        has the same False Positive Error Rate. For example, if race has false positive parity, it implies that all three 
        races have the same False Positive Error Rate.\n\n**When should I care about False Positive Parity?** If your desired outcome is to make false positive errors equally on 
        people from all races, then you care about this criteria. This is important in cases where your intervention is punitive 
        and has risk of adverse consequences for the selected set. Using this criteria allows you to make sure that 
        you’re not making mistakes about any single group disproportionately. """ + oneline
        mkdown_highlevel += '**The Bias Report has found that the following groups do not have False Positive Parity:**\n\n'
        if 'FPR Parity' in group_value_df.columns:
            mkdown_highlevel += '\n\n##### False Positive Rate\n\n'
            mkdown_highlevel += get_false_text(group_value_df, 'FPR Parity', fairness_measures_depend)
        if 'FDR Parity' in group_value_df.columns:
            mkdown_highlevel += '\n\n##### False Discovery Rate\n\n'
            mkdown_highlevel += get_false_text(group_value_df, 'FDR Parity', fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline
    if 'TypeII Parity' in group_value_df.columns:
        mkdown_highlevel += '\n\n### False Negative Parity\n\n'
        mkdown_highlevel += """**What is it?** This criteria considers an attribute to have False Negative parity if every group 
        has the same False Negative Error Rate. For example, if race has false negative parity, it implies that all three races 
        have the same False Negative Error Rate.\n\n**When should I care about False Negative Parity?** If your desired outcome is to make false negative errors equally on 
        people from all races, then you care about this criteria. This is important in cases where your intervention is 
        assistive and missing an individual could lead to adverse outcomes for them. Using this criteria allows you to make sure 
        that you’re not missing people from certain groups disproportionately.\n\n""" + oneline
        mkdown_highlevel += '**The Bias Report has found that the following groups do not have False Negative Parity:**\n\n'
        if 'FPR Parity' in group_value_df.columns:
            mkdown_highlevel += '\n\n##### False Negative Rate\n\n'
            mkdown_highlevel += get_false_text(group_value_df, 'FNR Parity', fairness_measures_depend)
        if 'FDR Parity' in group_value_df.columns:
            mkdown_highlevel += '\n\n##### False Omission Rate\n\n'
            mkdown_highlevel += get_false_text(group_value_df, 'FOR Parity', fairness_measures_depend) + oneline
        mkdown_highlevel += '\n\n[Go to Top](#)' + oneline

    mkdown_parity = '\n\n## Fairness Criteria Assessments' + oneline
    # do we want to show this?
    # mkdown_parity += get_highlevel_report(group_attribute_df) + '\n\n'

    mkdown_disparities = '\n\n## Some Numbers: Bias Metrics'
    mkdown_group = '\n\n## More Numbers: Group Metrics'
    # setup the group_value_df (colors and stuff)
    group_value_df = setup_group_value_df(group_value_df, configs.fair_measures_requested,
                                          fairness_measures_depend)
    for attr in configs.attr_cols:
        mkdown_parity += '\n\n### ' + attr + oneline
        mkdown_disparities += '\n\n### ' + attr + oneline
        mkdown_group += '\n\n### ' + attr + oneline
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

    report = mkdown_highlevel + '----' + mkdown_parity + '----' + mkdown_disparities + '----' + mkdown_group
    report_html = markdown(report, extras=['tables', 'header-ids'])
    # coloring True/False results
    report_html = report_html.replace('nan', 'Undefined')
    report_html = report_html.replace('>False<', ' style="color:red"><b>Unfair</b><')
    report_html = report_html.replace('>True<', ' style="color:green"><b>Fair</b><')

    report_html = report_html.replace('##br##', '<br>')
    report_html = report_html.replace('>##red##', ' style="color:red">')
    report_html = report_html.replace('>##green##', ' style="color:green">')

    report_html = report_html.replace(' unfair ', '<span style="color:red"><b> unfair </b></span>')
    report_html = report_html.replace(' fair ', '<span style="color:green"><b> fair </b></span>')

    report_html = report_html.replace('<table>', '<table class="table table-striped" >')
    # report_html = report_html.replace('< thead >\n < tr >', '< thead >\n < tr class="table-info" >')
    report_html = report_html.replace('<h1', '<br><h1 align="center" ')


    return report_html
