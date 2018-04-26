# Run aequitas piece by piece and look for discrepancies from 'groud_truth'


import pandas as pd
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from ..utils.configs_loader import Configs
#read test.csv
#read expected output


def helper(input_filename, expected_filename, config):
	'''

	'''	
	expected_df = pd.read_csv(expected_filename)
	
	test_df, _  = audit(pd.read_csv(input_filename), config)

	# match expected_df columns
	shared_columns = [c for c in expected_df.columns if c in test_df.columns]

	try:
		expected_df = expected_df[shared_columns]
		test_df = test_df[shared_columns]
		combined_data =  pd.merge(expected_df, test_df, on=['attribute_name','attribute_value'])
	# subtract expected_df from test_df 

	# see if close enough to 0

	return combined_data




