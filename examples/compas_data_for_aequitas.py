# The purpose of this script is to transform raw data to 
# the format expected by Aequitas.
#
# SOURCE: ProPublica 
# Data: https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv
# ProPublica's methodology: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
# Ari Anisfeld

import pandas as pd
df = pd.read_csv('./raw_data/compas-scores-two-years.csv')


# rename for tool
df = df.rename(columns={'id':'entity_id', 
                        'two_year_recid':'label_value'})

# score_text is 'High', 'Medium' or 'Low' and reflects level of assessed risk of recidivism
# "High" and "Medium" are considered prediction that the defendant is charged with a felony 
# or misdemenor in the two years after administation of COMPAS assessment. "Low" is considered 
# a prediction of non-recidivism. This is based on ProPublica's interpretation of Northpointe's
# practioner guide.
#
# "According to Northpointe’s practitioners guide, COMPAS “scores in the medium and high range 
# garner more interest from supervision agencies than low scores, as a low score would suggest
# there is little risk of general recidivism,” so we considered scores any higher than “low” to 
# indicate a risk of recidivism."
# (https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)

df.loc[df['score_text'] == 'Low', 'score'] = str(0.0)
df.loc[df['score_text'] != 'Low', 'score'] = str(1.0)


df = df[['entity_id', 'score', 'label_value', 'race', 'sex', 'age_cat']]


df.to_csv('./data/compas_for_aequitas.csv', index=False)