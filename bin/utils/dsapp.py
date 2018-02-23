import pandas as pd
import logging
from sys import exit
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


def get_dsapp_data(engine, model_id, attrib_query, predictions_table):
    """

    :param engine:
    :param model_id:
    :param attrib_query:
    :param predictions_table:
    :return:
    """
    prediction_query = get_prediction_query(predictions_table, model_id)
    results_query = """
            with prediction_table as (
            {prediction_query}
            ), group_data as (
            {attributes_query}
            )
            SELECT * FROM prediction_table LEFT JOIN group_data USING (entity_id);    
        """.format(prediction_query=prediction_query, attributes_query=attrib_query)
    try:
        df = pd.read_sql(results_query, engine)
    except SQLAlchemyError:
        logging.error('PG: could not get dsapp predictions-attributes join table.')
    return df


def get_engine(configs):
    """

    :param configs:
    :return: Returns a SQLAlchemy pg connnection.
    """

    try:
        db_access = configs['dsapp']['db_credentials']
    except KeyError:
        logging.error('KeyError in configuration file.')
        exit()
    try:
        engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{db}'.format(
            user=db_access['user'],
            pw=db_access['password'],
            host=db_access['host'],
            port=db_access['port'],
            db=db_access['database']))
    except SQLAlchemyError:
        logging.error('PG: could not create pg engine!')
        exit()
    return engine


def get_models(configs, engine):
    """

    :param configs:
    :param engine:
    :return: Returns the list of models to search for in the predictions table.
    """

    try:
        models_query = configs['dsapp']['models_query']
    except KeyError:
        logging.error('Configs: could not load models_query (dsapp section)')
    try:
        models = pd.read_sql(models_query, engine)
    except SQLAlchemyError:
        logging.error('PG: models query failed!')
    return list(models['model_id'])


def get_prediction_query(predictions_table, model_id):
    """

    :param predictions_table:
    :param model_id:
    :return:
    """
    prediction_query = """
    WITH sub AS (SELECT
             model_id,
             as_of_date,
             entity_id,
             score,
             label_value,
             row_number()
             OVER (
               PARTITION BY model_id
               ORDER BY score DESC )     AS rank_abs,
             count(*)
             OVER (
               PARTITION BY model_id )  total_count
           FROM {predictions_table}
           WHERE model_id = '{model_id}') SELECT
           model_id, 
           as_of_date, 
           entity_id, score,
           label_value, 
           rank_abs,
           100.0 * (rank_abs::FLOAT / total_count) AS rank_pct 
           FROM sub
    """.format(predictions_table=predictions_table, model_id=model_id)
    return prediction_query


def create_bias_tables(db_engine):
    """

    :param db_engine:
    :return: Convenience function to run the CREATE TABLE statements
    """
    print('Creating aequitas tables....')
    query = '''
           DROP TABLE IF EXISTS results.aequitas_group;
           CREATE TABLE results.aequitas_group (
               model_id INTEGER,
               as_of_date TIMESTAMP,
               threshold_value FLOAT, -- for example, 1 or 0.5
               threshold_unit TEXT, -- should be 'pct' or 'abs'
               parameter TEXT, -- should be 'pct' or 'abs'
               k INTEGER,
               group_variable TEXT, -- the protected status, like 'gender'
               group_value TEXT,  -- like 'female'
               pp_k REAL,
               pn_k REAL,
               ppr_k REAL,
               ppr_g REAL,
               tp REAL,
               fp REAL,
               tn REAL,
               fn REAL, 
               tpr REAL,
               fpr REAL,
               tnr REAL,
               fnr REAL,
               precision REAL,
               fdr REAL,
               fomr REAL,
               npv REAL
           );
           DROP TABLE IF EXISTS results.aequitas_priors;
           CREATE TABLE results.aequitas_priors (
               model_id INTEGER,
               as_of_date TIMESTAMP,
               group_variable TEXT, -- the protected status, like 'gender'
               group_value TEXT,  -- like 'female'
               group_size INTEGER, -- the number of entities of this group_variable and group_value
               group_label_pos INTEGER,
               group_label_neg INTEGER,
               total_entities INTEGER -- the total number of entities on this predictions list
               );
           '''
    db_engine.execute(query)


