========
Aequitas
========

----------------------------------------------
The machine learning Bias and Fairness library
----------------------------------------------

An open-source bias audit toolkit for machine learning developers, analysts, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive risk-assessment tools.

Demo
====

See what Aequitas can do at http://aequitas.dssg.io/.

Installation
============

Aequitas requires Python 3.

Install this Python library from source::

    python setup.py install

...or named as an installation requirement, *e.g.* via ``pip``::

    pip install git+https://github.com/dssg/aequitas.git

You may then import the ``aequitas`` module from Python::

    import aequitas

...or execute the auditor from the command line::

    aequitas-report

Development
===========

Provision your development environment via ``install``::

    ./install

Common development tasks, such as deploying the webapp, may then be handled via ``manage``::

    manage --help
