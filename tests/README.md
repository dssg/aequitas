# Unit Testing in Aequitas

To run unittests locally, you need to run the following commands from the base folder of the project.

The below commans only needs to be run once to give the testing script necessary permission to run

`chmod +x run_tests.sh`

The testing script itself can be run using the following commans 

`./run_tests.sh`

The script will run all defined tests in a virtual environment and output the coverage report in the terminal. It will also add an xml file of the coverage report `cov.xml` to the base folder.
