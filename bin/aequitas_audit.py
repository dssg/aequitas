# change relative imports
from bin.utils.dsapp import get_dsapp_data
from bin.utils.dsapp import get_engine
from bin.utils.dsapp import get_models
from bin.utils.dsapp import create_bias_tables
from src.aequitas.group import Group

from sys import exit
import logging
import argparse
import yaml
import pandas as pd

about = """
##########################################################################
##   Center for Data Science and Public Policy                          ##
##   http://dsapp.uchicago.edu                                          ##
##                                                                      ##
##   Copyright 2017. The University of Chicago. All Rights Reserved.    ## 
##########################################################################



_________________________________________________________________________
                                     _ _            
               __ _  ___  __ _ _   _(_) |_ __ _ ___ 
              / _` |/ _ \/ _` | | | | | __/ _` / __| 
             | (_| |  __/ (_| | |_| | | || (_| \__ \ 
              \__,_|\___|\__, |\__,_|_|\__\__,_|___/
                            |_|                            
_________________________________________________________________________  



"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=about + 'Runs audits on predictions of machine learning models '
                            'to calculate a variety of bias and fairness metrics.\n')

    parser.add_argument('--format',
                        action='store',
                        dest='format',
                        default='csv',
                        type=str,
                        help='Data input format: csv or pg (postgres db).')

    parser.add_argument('--score-prediction',
                        action='store',
                        dest='score_predict',
                        default='score',
                        type=str,
                        help='Data input format: csv or pg (postgres db).')

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

    return parser.parse_args()


def run_dsapp(engine, configs):
    """

    :param engine:
    :param configs:
    :return:
    """
    print('run_dsapp()')
    try:
        thresholds = configs['thresholds']
        predictions_table = configs['dsapp']['predictions_table']
        attrib_query = configs['dsapp']['attributes_query']
        models = get_models(configs, engine)

    except KeyError:
        logging.error('KeyError in configuration file (dsapp section).')
    engine = get_engine(configs)
    g = Group(thresholds)
    count = 0
    for model_id in models:
        count += 1
        print(count)
        df = get_dsapp_data(engine, model_id, attrib_query, predictions_table)
        results, priors = g.get_crosstabs(df, thresholds, push_to_db=False)
    return



def main():
    args = parse_args()
    if args.format not in ['csv', 'dsapp']:
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
    if args.format == 'dsapp':
        engine = get_engine(configs)
        if args.create_tables:
            create_bias_tables(db_engine=engine)
        run_dsapp(engine, configs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
