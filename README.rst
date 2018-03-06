======
aequitas
======

Bias and Fairness audit tool.

.. image:: https://github.com/dssg/aequitas-public/blob/master/bin/utils/aequitas_report_header.png

Runs audits on predictions of machine learning models to calculate a variety of bias metrics, combines the results, and displays a report to select tradeoffs between performance and bias.

## Requirements
Requires the following information:
1. prediction scores gvien by a model
2. true labels for those examples
3. performance metrics of interest
4. bias metrics of interest
5. variables to calculate bias for (gender, race, ethnicity, etc.)

Currently we support the DSAPP schemas for the information above thart contain the following tables:
1. predictions
2. evaluations
3. models
4. model_groups

The output gets stored in the bias_raw and bias_complete tables.


## Usage

1. Create db_credentials.yaml file inside bias/

``
cd aequitas/


vim bias/db_credentials.yaml
    host: xxxx
    database: xxxx
    user: xxxx
    password: xxxx
    port: 5432
``

2. Edit PG queries in bias/sql/project_queries.py to make it compliant with your project's schema and the protected variables you want to perform bias analyses on.

3. Edit bias/configs.py to define the train_end_time of the models you want to use on the bias analysis. You can also define the thresholds to calculate metrics based on your results.evaluations table.

4. To create and populate bias tables (bias_raw and bias_complete) run the following in aequitas/:

``python3 -m bias.create_bias_tables``

