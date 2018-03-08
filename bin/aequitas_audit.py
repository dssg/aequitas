
import argparse
import logging
from sys import exit

import yaml

from bin.utils.db import create_bias_tables
from bin.utils.db import get_dsapp_data
from bin.utils.db import get_engine
from bin.utils.db import get_models
from bin.utils.report import audit_report
from bin.utils.report import get_group_value_report
from src.aequitas.bias import Bias
from src.aequitas.fairness import Fairness
from src.aequitas.group import Group

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

    parser.add_argument('--io-format',
                        action='store',
                        dest='format',
                        default='csv',
                        type=str,
                        help='Data input/output format: csv or db (postgres db).')

    parser.add_argument('--ref_group',
                        action='store',
                        dest='ref_groups',
                        default='predefined',
                        type=str,
                        help='Reference group method for bias metrics: min_metric, majority, '
                             'predefined')

    parser.add_argument('--config',
                        action='store',
                        dest='config_file',
                        default='config.yaml',
                        help='Absolute filepath for input yaml config file.')

    parser.add_argument('--input',
                        action='store',
                        dest='input_file',
                        default='',
                        help='Absolute filepath for input dataset in csv format.')

    parser.add_argument('--output-folder',
                        action='store',
                        dest='output_folder',
                        default='',
                        help='Folder name to be created inside aequitas/output/')

    parser.add_argument('--create_tables',
                        action='store_true',
                        dest='create_tables',
                        default=False,
                        help='Create aequitas tables from scratch. Drop existing tables.')

    parser.add_argument('--score-prediction',
                        action='store',
                        dest='score_predict',
                        default='score',
                        type=str,
                        help='Data input format: score or prediction')

    return parser.parse_args()


def run_db(engine, configs, ref_groups_method):
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
    groups_model_list = []
    attributes = []
    for model_id in models:
        print('\n\n\nMODEL_ID: ', model_id)
        count += 1
        print(count)
        df = get_dsapp_data(engine, model_id, attrib_query, predictions_table)
        print(df.head(1))
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
        parameter = '300_abs'
        # fair_results = {'Overall Fairness': False}
        model_eval = 'xx.yy'
        group_value_report = get_group_value_report(group_value_df)
        audit_report(model_id, parameter, attributes, model_eval, configs, fair_results,
                     f.fair_measures,
                     ref_groups_method, group_value_report)
    return



def main():
    args = parse_args()
    print(about)
    if args.format not in {'csv', 'db'}:
        logging.error('Please define input data --format: csv or dsapp (postgres db with DSAPP '
                      'schemas)')
        exit()
    try:
        with open(args.config_file) as f:
            configs = yaml.load(f)
    except FileNotFoundError:
        logging.error('Could not load configurations! Please set configs.yaml file using --config')
        exit()
    if not configs:
        logging.error('Empty configurations! Please set configs.yaml file using --config')
        exit()

    # when having score vs prediction compatibility, thresholds just make sense for score
    if args.format == 'db':
        engine = get_engine(configs)
        if args.create_tables:
            create_bias_tables(engine, args.output_schema)
        run_db(engine, configs, args.ref_groups)


"""
        if push_to_db:
            groups_df.set_index(['model_id', 'as_of_date', 'group_variable']).to_sql(
                schema='results',
                name='bias_raw',
                con=db_conn,
                if_exists='append')
            priors_df.set_index(['model_id', 'as_of_date']).to_sql(schema='results', name='priors_df',
                                                                con=db_conn,
                                                                if_exists='append')
        if push_to_file:
            groups_df.to_csv('group_metrics.csv', sep='\t', encoding='utf-8')
            priors_df.to_csv('priors_df.csv', sep='\t', encoding='utf-8')
"""



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
