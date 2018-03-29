
import argparse
import logging
from sys import exit

import pandas as pd
import yaml

from bin.utils.io import get_csv_data
from bin.utils.io import get_db_data
from bin.utils.io import get_engine
from bin.utils.io import push_tocsv
from bin.utils.io import push_todb
from bin.utils.report import audit_report
from src.aequitas.bias import Bias
from src.aequitas.fairness import Fairness
from src.aequitas.group import Group

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

    parser.add_argument('--ref-group',
                        action='store',
                        dest='ref_groups',
                        default='majority',
                        type=str,
                        help='Reference group method for bias metrics: min_metric, majority, '
                             'predefined')

    parser.add_argument('--create-report',
                        action='store_true',
                        dest='report',
                        default=False,
                        help='If --report, then a pdf report is produced and stored in the output directory.')


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


def audit(df, ref_groups_method='majority', model_id=1, configs=None, report=True):
    """

    :param df:
    :param ref_groups_method:
    :param model_id:
    :param configs:
    :param report:
    :return:
    """
    g = Group()
    if configs:
        thresholds = configs['thresholds'] if 'thresholds' in configs else None
    else:
        thresholds = None
    groups_model, attributes = g.get_crosstabs(df, thresholds, model_id)
    print('df shape from the crosstabs:', groups_model.shape)
    b = Bias()
    if ref_groups_method == 'predefined' and 'reference_groups' in configs:
        bias_df = b.get_disparity_predefined_groups(groups_model, configs['reference_groups'])
    elif ref_groups_method == 'majority':
        bias_df = b.get_disparity_major_group(groups_model)
    else:
        bias_df = b.get_disparity_min_metric(groups_model)
    print('number of rows after bias majority ref group:', len(bias_df))
    print('Any NaN?: ', bias_df.isnull().values.any())
    print('df shape after bias minimum per metric ref group:', bias_df.shape)
    f = Fairness()
    group_value_df = f.get_group_value_fairness(bias_df)
    print('_______________\nGroup Value level:')
    print(group_value_df)
    group_variable_df = f.get_group_variable_fairness(group_value_df)
    print('_______________\nGroup Variable level:')
    print(group_variable_df)
    fair_results = f.get_overall_fairness(group_variable_df)
    print('_______________\nModel level:')
    print(fair_results)
    parameter = 'xyz_abs'
    # fair_results = {'Overall Fairness': False}
    model_eval = 'xx.yy'
    if report is True:
        audit_report(model_id, parameter, attributes, model_eval, configs, fair_results,
                     f.fair_measures,
                     ref_groups_method, group_value_df)
    return group_value_df


def run(df, ref_groups_method, configs, report):
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
            for model_id in df.model_id.unique():
                model_df = audit(df, ref_groups_method=ref_groups_method, model_id=model_id, configs=configs, report=report)
                model_df_list.append(model_df)
            group_value_df = pd.concat(model_df_list)

        else:
            group_value_df = audit(df, ref_groups_method=ref_groups_method, configs=configs, report=report)
    else:
        logging.error('run_csv: could not load a proper dataframe from the input filepath provided.')
        exit(1)
    return group_value_df

def main():
    args = parse_args()
    print(about)
    configs = None
    try:
        with open(args.config_file) as f:
            configs = yaml.load(f)
    except FileNotFoundError:
        logging.error('Could not load configurations! Please set configs.yaml file using --config')
        exit(1)
    if args.input_file is None:
        if configs is None:
            logging.error('No input file provided, so I assume you want to connect to a db, wait... but you also forget to '
                          'provide db credentials in the configs yaml file...! ')
            exit(1)
        engine = get_engine(configs)
        output_schema = 'public'
        if 'output_schema' in configs['db']:
            output_schema = configs['db']['output_schema']
        create_tables = 'append'
        if args.create_tables:
            create_tables = 'replace'
        input_query = configs['db']['input_query']
        df = get_db_data(engine, input_query)
        group_value_df = run(df, args.ref_groups, configs, args.report)
        push_todb(engine, output_schema, create_tables, group_value_df)
    else:
        df = get_csv_data(args.input_file)
        group_value_df = run(df, args.ref_groups, configs, args.report)
        push_tocsv(args.input_file, args.output_folder, group_value_df)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
