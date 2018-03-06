======
aequitas
======

Bias and Fairness audit tool.


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


