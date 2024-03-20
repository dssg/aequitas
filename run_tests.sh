#!/bin/bash

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r ./requirements/main.txt -r ./requirements/cli.txt -r ./requirements/webapp.txt
pip install -e .

pip install pytest pytest-cov

# Run the tests
coverage run -m pytest --cov-report xml:cov.xml --cov-report term

# Deactivate the virtual environment
deactivate
