# change relative imports
from ..src.aequitas.dsapp import create_bias_tables
from ..src.aequitas.dsapp import get_models
from ..src.aequitas.dsapp import get_queries

import logging
import argparse
import sys
import yaml
from sqlalchemy import create_engine

about = """
Center for Data Science and Public Policy
dsapp.uchicago.edu

Copyright 2017.  The University of Chicago. All Rights Reserved.  

_______________________________________________________
                                _ _            
          __ _  ___  __ _ _   _(_) |_ __ _ ___ 
         / _` |/ _ \/ _` | | | | | __/ _` / __| 
        | (_| |  __/ (_| | |_| | | || (_| \__ \ 
         \__,_|\___|\__, |\__,_|_|\__\__,_|___/
                       |_|                            
_______________________________________________________  



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
    try:
        models_table = configs['table_results_models']
        prediction_table = configs['table_results_predictions']
        protected_attr = configs['protected_attributes']
        thresholds = configs['thresholds']
    except KeyError:
        logging.error('KeyError in configuration file (dsapp section).')

    exp_query, model_query, prediction_query = get_queries(models_table, prediction_table)

    for row in models.iterrows():
        count += 1
        print(count)
        model_id = row[1]['model_id']
        as_of_date = row[1]['as_of_date']
        # g.get_crosstabs(model_id,...)


def main():
    args = parse_args()
    if args.format not in ['csv', 'dsapp']:
        logging.error('Please define input data --format: csv or dsapp (postgres db with DSAPP '
                      'schemas)')
        sys.exit()
    try:
        with open('../' + args.config_file) as f:
            configs = yaml.load(f)
    except FileNotFoundError:
        logging.error('Could not load configurations! Please set configs.yaml file using --config')
        sys.exit()
    if not configs:
        logging.error('Empty configurations! Please set configs.yaml file using --config')
        sys.exit()

    # when having score vs prediction compatibility, thresholds just make sense for score
    thresholds = configs['thresholds']
    if args.format == 'dsapp':
        try:
            db_access = configs['db_credentials']
            engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{db}'.format(
                user=db_access['user'],
                pw=db_access['password'],
                host=db_access['host'],
                port=db_access['port'],
                db=db_access['database']))
        except KeyError:
            logging.error('KeyError in configuration file.')
            sys.exit()
        if args.create_tables:
            create_bias_tables(db_engine=engine)

        run_dsapp(engine, configs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
