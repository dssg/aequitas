
import argparse
import logging
from datetime import datetime
from sys import exit

import pandas as pd
import yaml

from bin.utils.db import create_bias_tables
from bin.utils.db import get_db_data
from bin.utils.db import get_engine
from bin.utils.db import get_models
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
    # TODO
    parameter = '300_abs'
    # fair_results = {'Overall Fairness': False}
    model_eval = 'xx.yy'
    #TODO
    if report is True:
        audit_report(model_id, parameter, attributes, model_eval, configs, fair_results,
                     f.fair_measures,
                     ref_groups_method, group_value_df)
    return group_value_df


def run_db(engine, ref_groups_method, configs, report):
    """

    :param engine:
    :param configs:
    :return:
    """
    print('run_dsapp()')
    models = None
    try:
        thresholds = configs['thresholds']
        predictions_table = configs['db']['predictions_table']
        attrib_query = configs['db']['attributes_query']
        models = get_models(configs, engine)
    except KeyError:
        logging.error('KeyError in configuration file (db section).')
    engine = get_engine(configs)
    g = Group()
    count = 0
    model_df_list = []
    for model_id in models:
        print('\n\n\nMODEL_ID: ', model_id)
        count += 1
        print(count)
        df = get_db_data(engine, model_id, attrib_query, predictions_table)
        print(df.head(1))
        model_df = audit(df, ref_groups_method=ref_groups_method, model_id=model_id, configs=configs, report=report)
        model_df_list.append(model_df)
    group_value_df = pd.concat(model_df_list)
    group_value_df.set_index(['model_id', 'group_variable']).to_sql(
        schema='results',
        name='aequitas_groups',
        con=engine,
        if_exists='append')
    return


def run_csv(input_file, ref_groups_method, configs, report):
    """

    :param input_file:
    :param ref_groups_method:
    :param configs:
    :param report:
    :return:
    """
    print('run_csv()')
    df = None
    try:
        df = pd.read_csv(input_file)
    except IOError:
        logging.error('run_csv: could not load csv provided as input.')
        exit()
    if df is not None:
        if 'model_id' in df.columns:
            model_df_list = []
            for model_id in df.model_id.unique():
                model_df = audit(df, ref_groups_method=ref_groups_method, model_id=model_id, configs=configs, report=report)
                model_df_list.append(model_df)
            group_value_df = pd.concat(model_df_list)

        else:
            group_value_df = audit(df, ref_groups_method=ref_groups_method, configs=configs, report=report)
        datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        outpath = input_file[:input_file.find('.')] + '_aequitas_' + datestr + '.csv'
        group_value_df.to_csv(outpath, sep='\t', encoding='utf-8')
    else:
        logging.error('run_csv: could not load a proper dataframe from the input filepath provided.')
        exit()
    return

def main():
    args = parse_args()
    print(about)
    configs = None
    try:
        with open(args.config_file) as f:
            configs = yaml.load(f)
    except FileNotFoundError:
        logging.error('Could not load configurations! Please set configs.yaml file using --config')
        exit()
    if configs is None:
        logging.error('Empty configurations! Please set configs.yaml file using --config')
        exit()
    # when having score vs prediction compatibility, thresholds only make sense for score
    if args.input_file is None:
        engine = get_engine(configs)
        if args.create_tables:
            create_bias_tables(engine, args.output_schema)
        run_db(engine, args.ref_groups, configs, args.report)
    else:
        run_csv(args.input_file, args.ref_groups, configs, args.report)

"""
        if push_to_db:
            groups_df.set_index(['model_id', 'group_variable']).to_sql(
                schema='results',
                name='aequitas_groups',
                con=db_conn,
                if_exists='append')
            priors_df.set_index(['model_id']).to_sql(schema='results', name='priors_df',
                                                                con=db_conn,
                                                                if_exists='append')
        if push_to_file:
            groups_df.to_csv('group_metrics.csv', sep='\t', encoding='utf-8')
            priors_df.to_csv('priors_df.csv', sep='\t', encoding='utf-8')
"""



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
