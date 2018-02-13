import pandas as pd

class Group(object):
    def __init__(self, thresholds):

        self.label_neg_count = lambda label_col: lambda x: \
            (x[label_col] == 0).sum()

        self.label_pos_count = lambda label_col: lambda x: \
            (x[label_col] == 1).sum()

        #Defining the bias functions using lambda to later be used with .apply in pandas
        self.divide = lambda x,y: x/y if y != 0 else 0.0

        self.predicted_pos_group = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide((x[rank_col] <= thres).sum(), len(x) + 0.0)

        self.predicted_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] <= thres).sum()

        self.predicted_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] > thres).sum()

        self.predicted_pos_ratio_k = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide((x[rank_col] <= thres).sum(), k + 0.0)

        self.predicted_pos_ratio_g = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide((x[rank_col] <= thres).sum(), len(x) + 0.0)

        self.false_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 1)).sum()

        self.false_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 0)).sum()

        self.true_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 0)).sum()

        self.true_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 1)).sum()

        self.fpr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(), (x[label_col] ==0).sum().astype(
                float))

        self.tnr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum() , (x[label_col] ==
                                                                             0).sum().astype(
                float))

        self.fnr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(),(x[label_col] == 1).sum().astype(
                float))

        self.tpr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(),(x[label_col] ==
                                                                             1).sum().astype(
                float))

        self.fomr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(),(x[rank_col] >
                                                                               thres).sum(
            ).astype(float))

        self.npv = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum(),(x[rank_col] > thres).sum().astype(
                float))

        self.precision = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(), (x[rank_col] <=
                                                                               thres).sum(
            ).astype(float))

        self.fdr = lambda rank_col, label_col, thres, k: lambda x: \
            self.divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(),(x[rank_col] <=
                                                                               thres).sum(
            ).astype(float))



        self.bias_functions = {'tpr': self.tpr,
                               'tnr': self.tnr,
                               'fomr': self.fomr,
                               'fdr': self.fdr,
                               'fpr': self.fpr,
                               'fnr': self.fnr,
                               'npv': self.npv,
                               'precision': self.precision,
                               'pp_k': self.predicted_pos_count,
                               'pn_k': self.predicted_neg_count,
                               'ppr_k': self.predicted_pos_ratio_k,
                               'ppr_g': self.predicted_pos_ratio_g,
                               'fp': self.false_pos_count,
                               'fn': self.false_neg_count,
                               'tn': self.true_neg_count,
                               'tp': self.true_pos_count}

        # the columns in the evaluation table and the thresholds we want to apply to them
        self.thresholds = thresholds







def create_crosstabs(self, db_conn, models, thresholds, quantizers,
                     prediction_table_query,
                     staging_data_query,
                     push_to_db=True):
    '''
    Calculate various bias functions and prior distributions per model_id and as_of_date, and
    return two dataframes - one with FP/NP/FN/TN counts per model_id, as_of_date, and protected
    status value, and one with prior counts.

    Args:
        models (pd.DataFrame): Must have columns 'model_id' and 'as_of_date'.
                               Only models from this dataframe will be processed.
        prediction_table_query (str): query to return staging prediction table,
                                      with model_id and as_of_date still undefined
        staging_data_query (str): query to return officer data table (example at top
                                  of module), with model_id and as_of_date still
                                  undefined
        bias_functions (dict): a dictionary of name:lambda functions that will be applied to each group;
                                 names serve to label the resulting columns. Check top of this module
                                 for the default.
        thresholds (dict): a dictionary of column:value pairs, where the column needs to exist in the
                           predictions table, and the values are the thresholds to be applied for
                           that column. Check top of this module for an example.
        quantizers (dict): a dictionary of name: lamdba functions which are applied to each floaty
                           column, and should turn that column categorical. Check top of module
                           for an example.
        push_to_db (bool): If True, then the resulting dataframes will be pushed to the PG DB.
    '''
    if not quantizers:
        quantizers = self.quantizers
    results_df = pd.DataFrame(columns=['model_id', 'as_of_date', 'threshold_parameter',
                                       'group_variable', 'group_value', 'metric', 'value'])
    prior_df = pd.DataFrame(columns=['model_id', 'as_of_date', 'group_variable', 'group_value',
                                     'count', 'n'])
    dfs = []
    prior_dfs = []
    count = 0
    for row in models.iterrows():
        count += 1
        print(count)
        model_id = row[1]['model_id']
        as_of_date = row[1]['as_of_date']
        results_query = '''
                with prediction_table as (
                {prediction_table}
                ), group_data as (
                {group_table}
                )
                SELECT DISTINCT on (entity_id) * FROM prediction_table LEFT JOIN group_data 
                USING (
                entity_id);    
            '''.format(prediction_table=prediction_table_query.format(
            model_id=model_id, as_of_date=as_of_date),
            group_table=staging_data_query.format(model_id=model_id,
                                                  as_of_date=as_of_date)
        )
        print(results_query)
        # grab the labelled data
        df = pd.read_sql(results_query, db_conn)
        model_cols = ['model_id', 'as_of_date', 'entity_id', 'score', 'rank_abs', 'rank_pct',
                      'label_value']
        # within each model and as_of_date, discretize the floaty features
        float_cols = df.columns[
            (df.dtypes != object) & (df.dtypes != str) & (~df.columns.isin(model_cols))]
        for col in float_cols:
            # transform floaty columns into categories
            for qname, quantizer in quantizers.items():
                if qname == '':
                    raise ValueError("Quantizer name cannot be empty")
                df[col + '_' + qname] = quantizer(df[col])
                df = df.drop(col, 1)
        # calculate the bias for these columns
        feat_cols = df.columns[~df.columns.isin(model_cols)]  # index of the columns that are
        # not default(model_cols), therefore represent the group variables!
        print("Feature Columns (Groups):", feat_cols.values)
        # for each group variable do
        for col in feat_cols:
            # find the priors
            col_group = df.fillna({col: 'nan'}).groupby(col)
            counts = col_group.entity_id.count()
            # distinct entities within group value
            this_prior_df = pd.DataFrame({
                'model_id': [model_id] * len(counts),
                'as_of_date': [as_of_date] * len(counts),
                'group_variable': [col] * len(counts),
                'group_value': counts.index.values,
                'group_label_pos': col_group.apply(self.label_pos_count(
                    'label_value')).values,
                'group_label_neg': col_group.apply(self.label_neg_count(
                    'label_value')).values,
                'group_size': counts.values,
                'total_entities': [len(df)] * len(counts)
            })
            # for each model_id and as_of_date the priors_df has length group_variables * group_values
            prior_dfs.append(this_prior_df)
            # we calculate the bias for two different types of thresholds (percentage ranks and absolute ranks)
            for thres_unit, thres_values in thresholds.items():
                for thres_val in thres_values:
                    flag = 0
                    k = (df[thres_unit] <= thres_val).sum()
                    # print (thres_unit, thres_val, k)
                    for name, func in self.bias_functions.items():
                        func = func(thres_unit, 'label_value', thres_val, k)
                        feat_bias = col_group.apply(func)

                        '''
                        if not 'nan' in df[col]:
                            #feat_bias = df.fillna({col: 'nan'}).groupby(col).apply(func)
                            feat_bias = col_group.apply(func)
                        else:
                            raise ValueError(
                                "%s cannot be na-filled because the value 'nan' is already used" % col)

                        '''
                        metrics_df = pd.DataFrame({
                            'model_id': [model_id] * len(feat_bias),
                            'as_of_date': [as_of_date] * len(feat_bias),
                            'threshold_value': thres_val,
                            'threshold_unit': thres_unit[-3:],
                            'parameter': str(thres_val) + '_' + thres_unit[-3:],
                            'k': k,
                            'group_variable': [col] * len(feat_bias),
                            'group_value': feat_bias.index.values,
                            name: feat_bias.values
                        })
                        if flag == 0:
                            bias_df = metrics_df
                            flag = 1
                        else:
                            bias_df = bias_df.merge(metrics_df)
                        # print(bias_df.head(1))
                    dfs.append(bias_df)
    # precision@	25_abs
    results_df = pd.concat(dfs)
    priors = pd.concat(prior_dfs)
    # print(len(results_df), len(priors))
    if push_to_db:
        results_df.set_index(['model_id', 'as_of_date', 'group_variable']).to_sql(
            schema='results',
            name='bias_raw',
            con=db_conn,
            if_exists='append')
        priors.set_index(['model_id', 'as_of_date']).to_sql(schema='results', name='priors',
                                                            con=db_conn,
                                                            if_exists='append')

    return results_df, priors
