import logging

from markdown2 import markdown
from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True

logging.getLogger(__name__)

# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.


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
            aux_df.at[idx, col] = 'Ref'

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


def audit_report_markdown(configs, group_value_df, group_attribute_df, fairness_measures_depend, overall_fairness, model_id=1):
    manylines = '  \n&nbsp;\n\n      \n&nbsp;\n\n'
    oneline = '  \n&nbsp;\n\n'
    mkdown_highlevel = '    \n&nbsp;\n\n## Fairness Overview' + oneline
    mkdown_highlevel += get_highlevel_report(group_attribute_df) + manylines
    mkdown_highlevel += oneline + '### Table of Contents:\n'
    mkdown_highlevel += oneline + '1. [Fairness Measures Results](#fairness-measures-results) \n'
    mkdown_highlevel += '2. [Bias Metrics Results](#bias-metrics-results) \n'
    mkdown_highlevel += '3. [Group Metrics Results](#group-metrics-results) \n' + manylines

    mkdown_parity = '  \n&nbsp;\n\n## Fairness Measures Results'
    mkdown_disparities = '  \n&nbsp;\n\n## Bias Metrics Results'
    mkdown_group = '  \n&nbsp;\n\n## Group Metrics Results'
    # setup the group_value_df (colors and stuff)
    group_value_df = setup_group_value_df(group_value_df, configs.fair_measures_requested,
                                          fairness_measures_depend)
    for attr in configs.attr_cols:
        mkdown_parity += '  \n&nbsp;\n\n### ' + attr + oneline
        mkdown_disparities += '  \n&nbsp;\n\n### ' + attr + oneline
        mkdown_group += '  \n&nbsp;\n\n### ' + attr + oneline
        mkdown_parity += get_parity_group_report(group_value_df, attr, configs.fair_measures_requested, fairness_measures_depend)
        mkdown_disparities += get_disparities_group_report(group_value_df, attr, configs.fair_measures_requested,
                                                           fairness_measures_depend)
        mkdown_group += get_group_group_report(group_value_df, attr, configs.fair_measures_requested,
                                               fairness_measures_depend)
        mkdown_parity += '[Go to Top](#)' + manylines
        mkdown_disparities += '[Go to Previous]' + '(#' + '-'.join(attr.lower().split(' ')) + \
                              ')' + '\n\n' + '[Go to Top](#)' + manylines
        mkdown_group += '[Go to Previous]' + '(#' + '-'.join(attr.lower().split(' ')) + \
                        '-2)' + '\n\n' + '[Go to Top](#)' + manylines

    report = mkdown_highlevel + '----' + mkdown_parity + '----' + mkdown_disparities + '----' + mkdown_group
    report_html = markdown(report, extras=['tables', 'header-ids'])
    # coloring True/False results
    report_html = report_html.replace('>False', ' style="color:red"><b>Not Fair</b>')
    report_html = report_html.replace('>True', ' style="color:green"><b>Fair</b>')
    report_html = report_html.replace('>##red##', ' style="color:red">')
    report_html = report_html.replace('>##green##', ' style="color:green">')
    report_html = report_html.replace('<table>', '<table class="table">')

    return report_html


"""
def audit_report_pdf(model_id, parameter, model_eval, configs, fair_results, fair_measures,
                 group_value_df):

    group_value_report = get_group_value_report(group_value_df)
    project_description = configs.project_description
    if project_description is None:
        project_description = {'title': ' ', 'goal': ' '}
    if project_description['title'] is None:
        project_description['title'] = ' '
    if project_description['goal'] is None:
        project_description['goal'] = ' '
    print('\n\n\n:::::: REPORT ::::::\n')
    print('Project Title: ', project_description['title'])
    print('Project Goal: ', project_description['goal'])
    print('Bias Results:', str(fair_results))
    pdf = PDF()
    pdf.set_margins(left=20, right=15, top=10)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 5, project_description['title'], 0, 1, 'C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 10, datetime.now().strftime("%Y-%m-%d"), 0, 1, 'C')
    pdf.multi_cell(0, 5, 'Project Goal: ' + project_description['goal'], 0, 1)
    pdf.ln(2)
    model_metric = 'Precision at top ' + parameter
    pdf.multi_cell(0, 5, 'Model Perfomance Metric: ' + model_metric, 0, 1)
    pdf.multi_cell(0, 5, 'Fairness Measures: ' + ', '.join(fair_measures), 0, 1)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.multi_cell(0, 5, 'Model Audited: #' + str(model_id) + '\t Performance: ' + str(model_eval),
                   0, 1)
    pdf.set_font('Helvetica', '', 11)

    ref_groups = ''
    if configs.ref_groups_method == 'predefined':
        if configs.ref_groups:
            ref_groups = str(configs.ref_groups)
    elif configs.ref_groups_method == 'majority':
        ref_groups = ''
    elif configs.ref_groups_method == 'min_metric':
        ref_groups = ''
    else:
        logging.error('audit_report(): wrong reference group method!')
        exit(1)
    pdf.multi_cell(0, 5, 'Group attributes provided for auditing: ' + ', '.join(configs.attr_cols), 0, 1)
    pdf.multi_cell(0, 5, 'Reference groups used: ' + configs.ref_groups_method + ':   '
                                                                         '' + ref_groups, 0, 1)
    pdf.ln(4)
    results_text = 'aequitas has found that model #' + str(model_id) + ' is '
    if fair_results['Overall Fairness'] is True:
        is_fair = 'FAIR'
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, results_text + is_fair + '.', 0, 1)
    else:
        is_fair = 'UNFAIR'
        pdf.image('utils/img/danger.png', x=20, w=7, h=7)
        pdf.cell(73, 5, results_text, 0, 0)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 5, is_fair + '.', 0, 1)
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        for key, value in sorted(group_value_report.items()):
            pdf.ln(2)
            pdf.multi_cell(0, 5, value[0], 0, 1)
            pdf.ln(2)
            pdf.set_x(40.0)
            pdf.multi_cell(0, 5, '\t' + ', '.join(value[1]), 0, 1)
            pdf.ln(4)
            pdf.set_x(20.0)

    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_filename = 'aequitas_report_' + str(model_id) + '_' + project_description['title'].replace(' ',
                                                                                            '_') + '_' + datestr
    pdf.output('output/' + report_filename + '.pdf', 'F')
    return None
"""
