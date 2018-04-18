========
Aequitas
========

The machine learning Bias and Fairness library

An open-source bias audit toolkit for machine learning developers, analysts, and  policymakers to audit machine learning models for discrimination and bias, and make informed and equitable decisions around developing and deploying predictive risk-assessment tools.

Installation
============

Aequitas requires Python 3.

Install this Python library from source::

    python setup.py install

...or named as an installation requirement, *e.g.* via ``pip``::

    pip install git+https://github.com/dssg/aequitas.git

Use
===

You can use Aequitas in three ways:

1. Web App at http://aequitas.dssg.io

2. Python library: You may then import the ``aequitas`` module from Python::

    import aequitas

3. Execute the auditor from the command line::

    aequitas-report

Development
===========

Provision your development environment via ``install``::

    ./install

Common development tasks, such as deploying the webapp, may then be handled via ``manage``::

    manage --help
