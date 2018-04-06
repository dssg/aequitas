
import argparse
import logging
from sys import exit

import pandas as pd

from bin.utils.configs_loader import Configs
from bin.utils.io import get_csv_data
from bin.utils.io import get_db_data
from bin.utils.io import get_engine
from bin.utils.io import push_tocsv
from bin.utils.io import push_todb
from bin.utils.report import audit_report_markdown
from src.aequitas.bias import Bias
from src.aequitas.fairness import Fairness
from src.aequitas.group import Group
from src.aequitas.preprocessing import preprocess_input_df

# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.

about = """
############################################################################
##   Center for Data Science and Public Policy                            ##
##   http://dsapp.uchicago.edu                                            ##
##                                                                        ##
##   Copyright \xa9 2017. The University of Chicago. All Rights Reserved.    ## 
############################################################################



____________________________________________________________________________
                                     _ _            
               __ _  ___  __ _ _   _(_) |_ __ _ ___ 
              / _` |/ _ \/ _` | | | | | __/ _` / __| 
             | (_| |  __/ (_| | |_| | | || (_| \__ \ 
              \__,_|\___|\__, |\__,_|_|\__\__,_|___/
                            |_|                            

____________________________________________________________________________  

                    Bias and Fairness Analysis


"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=about + 'Runs audits on predictions of machine learning models '
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
                        default='configs/configs.yaml',
                        help='Absolute filepath for input yaml config file. Default is configs/configs.yaml')


    parser.add_argument('--output-folder',
                        action='store',
                        dest='output_folder',
                        default='output/',
                        help='Folder name to be created inside aequitas/output/')

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
    groups_model, attr_cols = g.get_crosstabs(df, score_thresholds=configs.score_thresholds, model_id=model_id,
                                              attr_cols=configs.attr_cols)
    print('audit: df shape from the crosstabs:', groups_model.shape)
    b = Bias()
    # todo move this to the new configs object / the attr_cols now are passed through the configs object...
    ref_groups_method = configs.ref_groups_method
    if ref_groups_method == 'predefined' and configs.ref_groups:
        bias_df = b.get_disparity_predefined_groups(groups_model, configs.ref_groups)
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model)
    else:
        bias_df = b.get_disparity_min_metric(groups_model)
    print('number of rows after bias majority ref group:', len(bias_df))
    print('Any NaN?: ', bias_df.isnull().values.any())
    print('bias_df shape:', bias_df.shape)
    f = Fairness(tau=configs.fairness_threshold)
    print('Fairness Threshold:', configs.fairness_threshold)
    print('Fairness Measures:', configs.fair_measures_requested)
    group_value_df = f.get_group_value_fairness(bias_df, fair_measures_requested=configs.fair_measures_requested)
    group_variable_df = f.get_group_variable_fairness(group_value_df, fair_measures_requested=configs.fair_measures_requested)
    print('_______________\nGroup Variable level:')
    print(group_variable_df)
    fair_results = f.get_overall_fairness(group_variable_df)
    print('_______________\nModel level:')
    print(fair_results)
    parameter = 'xyz_abs'
    model_eval = 'xx.yy'
    report = None
    if configs.report is True:
        report = audit_report_markdown(group_value_df, group_variable_df, overall_fairness=fair_results,
                                       fair_measures=configs.fair_measures_requested)
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
            report = report_list
        else:
            group_value_df, report = audit(df, configs=configs, preprocessed=preprocessed)
    else:
        logging.error('run_csv: could not load a proper dataframe from the input filepath provided.')
        exit(1)
    print(report)
    return group_value_df

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
        group_value_df = run(df, configs=configs, preprocessed=False)
        push_todb(engine, output_schema, create_tables, group_value_df)
    else:
        df = get_csv_data(args.input_file)
        group_value_df = run(df, configs=configs, preprocessed=False)
        push_tocsv(args.input_file, args.output_folder, group_value_df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
