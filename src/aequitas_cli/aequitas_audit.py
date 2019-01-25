import argparse
import logging
from sys import exit

import pandas as pd
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.preprocessing import preprocess_input_df

from .utils.configs_loader import Configs
from .utils.io import get_csv_data
from .utils.io import get_db_data
from .utils.io import get_engine
from .utils.io import push_tocsv
from .utils.io import push_todb
from .utils.io import push_topdf
from .utils.report import audit_report_markdown

__author__ = "Rayid Ghani, Pedro Saleiro <saleiro@uchicago.edu>, Loren Hinkson"
__copyright__ = "Copyright \xa9 2018. The University of Chicago. All Rights Reserved."

about = '\n'.join(
    [
        '############################################################################',
        '##   Center for Data Science and Public Policy                            ##',
        '##   http://dsapp.uchicago.edu                                            ##',
        '##                                                                        ##',
        '##   Copyright \xa9 2018. The University of Chicago. All Rights Reserved.    ##',
        '############################################################################',
        '____________________________________________________________________________',
        '',
        '                    ___                    _ __            ',
        '                   /   | ___  ____ ___  __(_) /_____ ______',
        '                  / /| |/ _ \/ __ `/ / / / / __/ __ `/ ___/',
        '                 / ___ /  __/ /_/ / /_/ / / /_/ /_/ (__  ) ',
        '                /_/  |_\___/\__, /\__,_/_/\__/\__,_/____/  ',
        '                              /_/    ',
        '',
        '',
        '',
        '____________________________________________________________________________',
        '',
        '                      Bias and Fairness Audit Tool                           ',
        '____________________________________________________________________________']
)

about2 = """
____________________________________________________________________________

                      Aequitas: Bias and Fairness Audit Tool
____________________________________________________________________________
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description=about2 + 'Runs audits on predictions of machine learning models '
                            'to calculate a variety of bias and fairness metrics.\n')

    parser.add_argument('--input',
                        action='store',
                        dest='input_file',
                        default=None,
                        help='Absolute filepath for input dataset in csv format. If no input is provided we assume there is a '
                             'db configuration in the configs.yaml file.')

    parser.add_argument('--config',
                        action='store',
                        dest='config_file',
                        default=None,
                        help='Absolute filepath for input yaml config file. Default is configs/configs.yaml')

    parser.add_argument('--output-folder',
                        action='store',
                        dest='output_folder',
                        default='',
                        help='Folder name to be created inside aequitas')

    parser.add_argument('--create-tables',
                        action='store_true',
                        dest='create_tables',
                        default=False,
                        help='Create aequitas table from scratch. Drop existing tables.')

    return parser.parse_args()


def audit(df, configs, model_id=1, preprocessed=False):
    """

    :param df:
    :param ref_groups_method:
    :param model_id:
    :param configs:
    :param report:
    :param preprocessed:
    :return:
    """
    if not preprocessed:
        df, attr_cols_input = preprocess_input_df(df)
        if not configs.attr_cols:
            configs.attr_cols = attr_cols_input
    g = Group()
    print('Welcome to Aequitas-Audit')
    print('Fairness measures requested:', ','.join(configs.fair_measures_requested))
    groups_model, attr_cols = g.get_crosstabs(df, score_thresholds=configs.score_thresholds, model_id=model_id,
                                              attr_cols=configs.attr_cols)
    print('audit: df shape from the crosstabs:', groups_model.shape)
    b = Bias()
    # todo move this to the new configs object / the attr_cols now are passed through the configs object...
    ref_groups_method = configs.ref_groups_method
    if ref_groups_method == 'predefined' and configs.ref_groups:
        bias_df = b.get_disparity_predefined_groups(groups_model, df, configs.ref_groups)
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model, df)
    else:
        bias_df = b.get_disparity_min_metric(groups_model, df)
    print('Any NaN?: ', bias_df.isnull().values.any())
    print('bias_df shape:', bias_df.shape)

    aqp = Plot()

    if len(configs.plot_bias_metrics) == 1:
        fig1 = aqp.plot_disparity(bias_df, metrics=configs.plot_bias_metrics)
    elif len(configs.plot_bias_metrics) > 1:
        fig1 = aqp.plot_disparity_all(bias_df, metrics=configs.plot_bias_metrics)
    if len(configs.plot_bias_disparities) == 1:
        fig2 = aqp.plot_group_metric(bias_df, metrics=configs.plot_bias_disparities)
    elif len(configs.plot_bias_disparities) > 1:
        fig2 = aqp.plot_group_metric_all(bias_df, metrics=configs.plot_bias_disparities)

    f = Fairness(tau=configs.fairness_threshold)
    print('Fairness Threshold:', configs.fairness_threshold)
    print('Fairness Measures:', configs.fair_measures_requested)
    group_value_df = f.get_group_value_fairness(bias_df, fair_measures_requested=configs.fair_measures_requested)
    group_attribute_df = f.get_group_attribute_fairness(group_value_df, fair_measures_requested=configs.fair_measures_requested)
    fair_results = f.get_overall_fairness(group_attribute_df)

    if len(configs.plot_bias_metrics) == 1:
        fig3 = aqp.plot_fairness_group(group_value_df, metrics=configs.plot_bias_metrics)
    elif len(configs.plot_bias_metrics) > 1:
        fig3 = aqp.plot_fairness_group_all(group_value_df, metrics=configs.plot_bias_metrics)

    if len(configs.plot_bias_disparities) == 1:
        fig4 = aqp.plot_fairness_disparity(group_value_df, metrics=configs.plot_bias_disparities)
    elif len(configs.plot_bias_metrics) > 1:
        fig4 = aqp.plot_fairness_disparity_all(group_value_df, metrics=configs.plot_bias_disparities)

    print(fair_results)
    report = None
    if configs.report is True:
        report = audit_report_markdown(configs, group_value_df, f.fair_measures_depend, fair_results)
    return group_value_df, report


def run(df, configs, preprocessed=False):
    """

    :param df:
    :param ref_groups_method:
    :param configs:
    :param report:
    :return:
    """
    group_value_df = None
    if df is not None:
        if 'model_id' in df.columns:
            model_df_list = []
            report_list = []
            for model_id in df.model_id.unique():
                model_df, model_report = audit(df.loc[df['model_id'] == model_id], model_id=model_id, configs=configs,
                                               preprocessed=preprocessed)
                model_df_list.append(model_df)
                report_list.append(model_report)
            group_value_df = pd.concat(model_df_list)
            report = '\n'.join(report_list)

        else:
            group_value_df, report = audit(df, configs=configs, preprocessed=preprocessed)
    else:
        logging.error('run_csv: could not load a proper dataframe from the input filepath provided.')
        exit(1)
    # print(report)
    return group_value_df, report


def main():
    args = parse_args()
    print(about)
    configs = Configs.load_configs(args.config_file)
    if args.input_file is None:
        if configs.db is None:
            logging.error('No input file provided, so I assume you want to connect to a db, wait... but you also forget to '
                          'provide db credentials in the configs yaml file...! ')
            exit(1)
        engine = get_engine(configs)
        if 'output_schema' in configs.db:
            output_schema = configs.db['output_schema']
        else:
            output_schema = 'public'
        create_tables = 'append'
        if args.create_tables:
            create_tables = 'replace'
        input_query = configs.db['input_query']
        df = get_db_data(engine, input_query)
        group_value_df, report = run(df, configs=configs, preprocessed=False)
        push_todb(engine, output_schema, create_tables, group_value_df)
    else:
        df = get_csv_data(args.input_file)
        group_value_df, report = run(df, configs=configs, preprocessed=False)
        push_tocsv(args.input_file, group_value_df)
        push_topdf(args.input_file, report)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
