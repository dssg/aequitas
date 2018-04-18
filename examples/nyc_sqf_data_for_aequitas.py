# The purpose of this script is to transform raw data to 
# the format expected by Aequitas.
#
# SOURCE: NYPD
# Data: http://www1.nyc.gov/assets/nypd/downloads/excel/analysis_and_planning/stop-question-frisk/sqf-2017.csv
# Webpage: http://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page
# Ari Anisfeld

import pandas as pd
df = pd.read_csv('./raw_data/sqf-2017.csv')


# rename for tool
df = df.rename(columns={'SUSPECT_RACE_DESCRIPTION':'race', 'SUSPECT_SEX':'sex'})

# One purported purpose of stop, question and frisk is to keep weapons off the street.
# We define a true positive as discovering a weapon and a false positive no discovering a weapon

df.loc[df['WEAPON_FOUND_FLAG'] == 'N', 'label_value'] = str(0.0)
df.loc[df['WEAPON_FOUND_FLAG'] == 'Y', 'label_value'] = str(1.0)

# Note the data do not observe negatives so all 'predictions' are positive
df['score']  = 1.0
#df.loc[df['FRISKED_FLAG'] == 'N', 'score'] = 0
#df.loc[df['FRISKED_FLAG'] == 'Y', 'score'] = 1


df = df.query('sex == "MALE" | sex == "FEMALE"').query('race != "MALE" & race != "(null)"')

df = df[['score', 'label_value', 'race', 'sex']]


df.to_csv('./data/nyc_sqf_for_aequitas.csv', index=False)