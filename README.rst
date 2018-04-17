========
Aequitas
========

The machine learning Bias and Fairness library


Installation
============

Aequitas requires Python 3.

Install this Python library from source::

    python setup.py install

...or named as an installation requirement, *e.g.* via ``pip``::

    pip install git+https://github.com/dssg/aequitas.git

Use
===

You may then import the ``aequitas`` module from Python::

    import aequitas

...and execute the auditor from the command line::

    aequitas-report

Development
===========

Provision your development environment via ``install``::

    ./install

Common development tasks, such as deploying the webapp, may then be handled via ``manage``::

    manage --help
