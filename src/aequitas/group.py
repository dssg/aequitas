import pandas as pd


class Group(object):
    def __init__(self):

        # the columns in the evaluation table and the thresholds we want to apply to them
        self.quantizers = {
            'quartiles': lambda s: 'quantile_' + pd.qcut(s, q=4, duplicates='drop', labels=False).
                map(lambda x: '%.0f' % x)}

        self.label_neg_count = lambda label_col: lambda x: \
            (x[label_col] == 0).sum()
        self.label_pos_count = lambda label_col: lambda x: \
            (x[label_col] == 1).sum()
        self.group_functions = self.get_group_functions()

    def get_group_functions(self):

        divide = lambda x, y: x / y if y != 0 else 0.0

        predicted_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] <= thres).sum()

        predicted_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            (x[rank_col] > thres).sum()

        predicted_pos_ratio_k = lambda rank_col, label_col, thres, k: lambda x: \
            divide((x[rank_col] <= thres).sum(), k + 0.0)

        predicted_pos_ratio_g = lambda rank_col, label_col, thres, k: lambda x: \
            divide((x[rank_col] <= thres).sum(), len(x) + 0.0)

        false_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 1)).sum()

        false_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 0)).sum()

        true_neg_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] > thres) & (x[label_col] == 0)).sum()

        true_pos_count = lambda rank_col, label_col, thres, k: lambda x: \
            ((x[rank_col] <= thres) & (x[label_col] == 1)).sum()

        fpr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(),
                   (x[label_col] == 0).sum().astype(
                       float))

        tnr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum(), (x[label_col] ==
                                                                         0).sum().astype(
                float))

        fnr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(),
                   (x[label_col] == 1).sum().astype(
                       float))

        tpr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(), (x[label_col] ==
                                                                          1).sum().astype(
                float))

        fomr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 1)).sum(), (x[rank_col] >
                                                                         thres).sum(
            ).astype(float))

        npv = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] > thres) & (x[label_col] == 0)).sum(),
                   (x[rank_col] > thres).sum().astype(
                       float))

        precision = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 1)).sum(), (x[rank_col] <=
                                                                          thres).sum(
            ).astype(float))

        fdr = lambda rank_col, label_col, thres, k: lambda x: \
            divide(((x[rank_col] <= thres) & (x[label_col] == 0)).sum(), (x[rank_col] <=
                                                                          thres).sum(
            ).astype(float))

        group_functions = {'tpr': tpr,
                           'tnr': tnr,
                           'fomr': fomr,
                           'fdr': fdr,
                           'fpr': fpr,
                           'fnr': fnr,
                           'npv': npv,
                           'precision': precision,
                           'pp_k': predicted_pos_count,
                           'pn_k': predicted_neg_count,
                           'ppr_k': predicted_pos_ratio_k,
                           'ppr_g': predicted_pos_ratio_g,
                           'fp': false_pos_count,
                           'fn': false_neg_count,
                           'tn': true_neg_count,
                           'tp': true_pos_count}
        return group_functions

    def get_crosstabs(self, df, thresholds,
                      push_to_db=False, push_to_file=False):
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
            group_functions (dict): a dictionary of name:lambda functions that will be applied to
            each group;
                                     names serve to label the resulting columns. Check top of this module
                                     for the default.
            push_to_db (bool): If True, then the resulting dataframes will be pushed to the PG DB.
        '''
        results_df = pd.DataFrame(columns=['model_id', 'as_of_date', 'threshold_parameter',
                                           'group_variable', 'group_value', 'metric', 'value'])
        prior_df = pd.DataFrame(columns=['model_id', 'as_of_date', 'group_variable', 'group_value',
                                         'count', 'n'])
        dfs = []
        prior_dfs = []

        model_cols = ['model_id', 'as_of_date', 'entity_id', 'score', 'rank_abs', 'rank_pct',
                      'label_value']
        # within each model and as_of_date, discretize the floaty features
        float_cols = df.columns[
            (df.dtypes != object) & (df.dtypes != str) & (~df.columns.isin(model_cols))]
        for col in float_cols:
            # transform floaty columns into categories
            for qname, quantizer in self.quantizers.items():
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
                    for name, func in self.group_functions.items():
                        func = func(thres_unit, 'label_value', thres_val, k)
                        feat_bias = col_group.apply(func)
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
        if push_to_file:
            results_df.to_csv('group_metrics.csv', sep='\t', encoding='utf-8')
            priors.to_csv('priors.csv', sep='\t', encoding='utf-8')
        return results_df, priors

    def get_group_metrics(self, df):
        return 0
