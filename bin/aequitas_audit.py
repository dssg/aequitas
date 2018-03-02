import argparse
import logging
from sys import exit

import pandas as pd
import yaml

from bin.utils.db import create_bias_tables
from bin.utils.db import get_dsapp_data
from bin.utils.db import get_engine
from bin.utils.db import get_models
from bin.utils.pdf_creator import PDF
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
                        dest='format',
                        default='min',
                        type=str,
                        help='Reference group method for bias metrics: min, major, predefined')

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
                        help='Data input format: csv or pg (postgres db).')

    return parser.parse_args()


def run_db(engine, configs):
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
    for model_id in models:
        print('MODEL_ID: ', model_id)
        count += 1
        print(count)
        df = get_dsapp_data(engine, model_id, attrib_query, predictions_table)
        print(df.head(1))
        groups_model = g.get_crosstabs(df, thresholds, model_id)
        groups_model_list.append(groups_model)
    group_results = pd.concat(groups_model_list, ignore_index=True)
    # print(group_results.head(10))
    # print(group_results[['model_id','parameter','k', 'group_variable','group_value','Prev','PPrev']])
    print('df shape from the crosstabs:', group_results.shape)
    b = Bias()
    bias_df = b.get_disparity_major_group(group_results)
    print('number of rows after bias majority ref group:', len(bias_df))
    print('Any NaN?: ', bias_df.isnull().values.any())
    """
    if 'reference_groups' in configs:
        bias_df = b.get_disparity_predefined_groups(group_results, configs['reference_groups'])

    """

    print('df shape after bias minimum per metric ref group:', bias_df.shape)
    f = Fairness()
    group_value_df = f.get_group_value_fairness(bias_df)
    print(group_value_df[['group_variable', 'group_value', 'parameter', 'FDR', 'FDR_disparity',
                          'FDR_ref_group_value', 'TypeI Parity', 'TypeII Parity', 'Impact Parity',
                          'Statistical Parity', 'Unsupervised Fairness', 'Supervised Fairness']])
    group_variable_df = f.get_group_variable_fairness(group_value_df)
    fair = f.get_overall_fairness(group_variable_df)
    audit_report(configs, fair)
    return


def audit_report(configs, fair_results):
    proj_desc = configs['project_description']
    print('\n\n\n:::::: REPORT ::::::\n')
    print('Project Title: ', proj_desc['title'])
    print('Project Goal: ', proj_desc['goal'])
    print('Bias Results:', str(fair_results))

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Project Title: ' + proj_desc['title'], 0, 1)
    pdf.cell(0, 10, 'Project Goal: ' + proj_desc['goal'], 0, 1)
    ref_groups = None
    if 'reference_groups' in configs:
        ref_groups = str(configs['reference_groups'])

    pdf.cell(0, 10, 'Reference Groups: ' + ref_groups, 0, 1)

    pdf.cell(0, 10, 'Fairness Results: ' + str(fair_results), 0, 1)

    # for i in range(1, 41):
    #    pdf.cell(0, 10, 'Printing line number ' + str(i), 0, 1)
    pdf.output('output/aequitas.pdf', 'F')

    return None

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
        run_db(engine, configs)


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
