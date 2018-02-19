import pandas as pd


def get_queries(models_table, prediction_table):
    exp_query = "SELECT %s FROM %s WHERE %s='{tr_end_time}'" % (models_table['model_id_column'],
                                                                models_table['table'], models_table[
                                                                    'train_end_time_column'])

    model_query = "SELECT DISTINCT %s, %s::TIMESTAMP AS as_of_date FROM %s" % ( \
        models_table['model_id_column'],
        models_table['train_end_time_column'],
        models_table['table'])

    prediction_query = "WITH t AS (SELECT %s, %s::TIMESTAMP AS as_of_date, %s AS entity_id, " \
                       "%s AS score, %s, row_number() OVER (PARTITION BY %s ORDER BY %s DESC ) AS " \
                       "rank_abs, count(*) OVER (PARTITION BY %s) group_count FROM %s LEFT JOIN %s " \
                       "USING (%s) WHERE %s='{model_id}' AND %s::TIMESTAMP='{as_of_date}') SELECT %s, " \
                       "as_of_date,  entity_id, score, %s, rank_abs, 100.0 * (" \
                       "rank_abs::FLOAT/group_count) " \
                       "AS rank_pct FROM t" % (prediction_table['model_id_column'],
                                               prediction_table['train_end_time_column'],
                                               prediction_table['entity_id_column'],
                                               prediction_table['score_column'],
                                               prediction_table['label_value_column'],
                                               prediction_table['model_id_column'],
                                               prediction_table['score_column'],
                                               prediction_table['model_id_column'],
                                               prediction_table['table'],
                                               models_table['table'],
                                               models_table['model_id_column'],
                                               models_table['model_id_column'],
                                               prediction_table['train_end_time_column'],
                                               prediction_table['model_id_column'],
                                               prediction_table['label_value_column'])

    return exp_query, model_query, prediction_query


def create_bias_tables(db_engine):
    ''' Convenience function to run the CREATE TABLE statements. '''
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


def get_models(db_conn, model_query,
               model_id_list=None, as_of_date_list=None):
    '''
    Returns a dataframe with columns model_id, as_of_date.

    model_query is a PG query that returns a table with model_id and as_of_date.
    (If defaults to one that works on the staging schema.)

    By default, grabs all model_ids and as_of_dates. If model_id_list and/or as_of_date_list
    is provided, then only rows where the respective value is an element of the
    list will be returned.
    '''

    if model_id_list or as_of_date_list:
        model_query += " WHERE "
    if model_id_list:
        model_query += " model_id=ANY('{%s}')" % ', '.join(map(str, model_id_list))
        if as_of_date_list:
            model_query += " AND "
    if as_of_date_list:
        model_query += " as_of_date=ANY('{%s}')" % ', '.join(map(str, as_of_date_list))
    return pd.read_sql(model_query, db_conn)
