# Aequitas
#
# Test code for aequitas_audit


import os
import sys

timeout = 60

sys.path.append(os.getcwd())

# Get the test files from the same directory as
# this file.
BASE_DIR = os.path.dirname(__file__)

import pytest
import pandas as pd
import numpy as np
from aequitas_cli.aequitas_audit import audit
from aequitas_cli.utils.configs_loader import Configs

def helper(input_filename, expected_filename, config_file):
	'''

	'''	

	input_filename = os.path.join(BASE_DIR, input_filename)
	expected_df = pd.read_csv(os.path.join(BASE_DIR, expected_filename))
	
	if config_file:
		config_file = os.path.join(BASE_DIR, config_file)

	config = Configs.load_configs(config_file)

	test_df, _  = audit(pd.read_csv(os.path.join(BASE_DIR, input_filename)), config)


	
	# match expected_df columns
	shared_columns = [c for c in expected_df.columns if c in test_df.columns]

	try:
		expected_df = expected_df[shared_columns]
		test_df = test_df[shared_columns]
		combined_data =  pd.merge(expected_df, test_df, on=['attribute_name','attribute_value'])
	# subtract expected_df from test_df 
	except:
		# collect output for 
		print('could not merge')
		return (test_df, expected_df)
	# see if close enough to 0

	s = ""
	EPS = 1e-6
	for col in shared_columns:
		if col not in {'attribute_value','attribute_name'}:
			print('testing {} ...'.format(col))

			try:
				if np.mean(combined_data[col + "_x"] - combined_data[col + "_y"]) > EPS:
						exp_mean = np.mean(combined_data[col + "_x"])
						aeq_mean = np.mean(combined_data[col + "_y"])
						s += "{} fails: Expected {} on average, but aequitas returned {}\n".format(col, exp_mean, aeq_mean)
						
						pytest.fail(s)
			except:
				if not all(combined_data[col + "_x"]==combined_data[col + "_y"]):
					s += "{} fails: at least one entry was not the same between data sets\n".format(col)
			
					pytest.fail(s)


# simplest tests
def test_group_class_1():
	# test that the results from group are as expected
	return helper('test_1.csv', 'expected_output_group_test_1.csv', None)

def test_bias_class_1():
	# test that the results from bias are as expected (note it also tests group)
	return helper('test_1.csv', 'expected_output_bias_test_1.csv', None)

def test_fairness_class_1():
	#test that the results from fairness are as expected (note it also tests bias and group)
	return helper('test_1.csv', 'expected_output_fairness_test_1.csv', None)


if __name__=='__main__':
	import pandas as pd
	#helper('./src/tests/test_1.csv', './src/tests/expected_output_test_1.csv', None)
	helper('test_1.csv', 'expected_output_test_1.csv', 'test_1.yaml')

