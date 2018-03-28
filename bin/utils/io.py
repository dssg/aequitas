import logging
from datetime import datetime
from os import path
from sys import exit

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


# Authors: Pedro Saleiro <saleiro@uchicago.edu>
#          Rayid Ghani
#
# License: Copyright \xa9 2018. The University of Chicago. All Rights Reserved.

def get_db_data(engine, input_query):
    """

    :param engine:
    :param input_query:
    :return:
    """
    logging.info('querying db...')
    try:
        df = pd.read_sql(input_query, engine)
    except SQLAlchemyError:
        logging.error('PG: could not get the resulting table. Please check your input_query.')
        exit(1)
    return df


def get_csv_data(input_file):
    """

    :param input_file:
    :return:
    """
    logging.info('loading csv data...')
    try:
        df = pd.read_csv(input_file)
    except IOError:
        logging.error('run_csv: could not load csv provided as input.')
        exit(1)
    return df


def push_todb(engine, output_schema, create_tables, output_df):
    """

    :param engine:
    :param output_df:
    :return:
    """
    logging.info('pushing to db aequitas table...')
    try:
        output_df.set_index(['model_id', 'group_variable']).to_sql(
            schema=output_schema,
            name='aequitas_group',
            con=engine,
            if_exists=create_tables)
    except SQLAlchemyError:
        logging.error('push_db_data: Could not push results to the target database.')
        exit(1)


def push_tocsv(input_file, output_folder, output_df):
    """

    :param input_file:
    :param output_folder:
    :param output_df:
    :return:
    """
    logging.info('pushing to csv...')
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    ipath, input_name = path.split(input_file)
    outpath = output_folder + input_name[:input_file.find('.')] + '_aequitas_' + datestr + '.csv'
    try:
        output_df.to_csv(outpath, encoding='utf-8', index=False)
    except IOError:
        logging.error('push_csv_data: Could not push results to a csv file.')
        exit(1)




def get_engine(configs):
    """

    :param configs:
    :return: Returns a SQLAlchemy pg connnection.
    """
    try:
        db_access = configs['db']['db_credentials']
    except KeyError:
        logging.error('KeyError in configuration file.')
        exit(1)
    try:
        engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{db}'.format(
            user=db_access['user'],
            pw=db_access['password'],
            host=db_access['host'],
            port=db_access['port'],
            db=db_access['database']))
    except SQLAlchemyError:
        logging.error('PG: could not create pg engine!')
        exit(1)
    return engine
