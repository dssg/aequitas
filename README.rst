========================================================
The Bias and Fairness Audit Toolkit
========================================================


.. figure:: src/aequitas_webapp/static/images/aequitas_header.png
   :scale: 50 %


--------
Aequitas
--------

Aequitas is an open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive risk-assessment tools.

`Learn more about the project <http://dsapp.uchicago.edu/aequitas/>`_.

Demo
====

`See what Aequitas can do <http://aequitas.dssg.io/>`_.

Sample Jupyter Notebook
=======================

`Explore bias analysis of the COMPAS data <https://github.com/dssg/aequitas/blob/master/docs/source/examples/compas_demo.ipynb>`_ using the Aequitas library.

Documentation
=============

Find documentation `here <https://dssg.github.io/aequitas/>`_.

Installation
============

Aequitas requires Python 3.

Install this Python library from source::

    python setup.py install

...or named as an installation requirement, *e.g.* via ``pip``::

    python -m pip install git+https://github.com/dssg/aequitas.git

You may then import the ``aequitas`` module from Python::

    import aequitas

...or execute the auditor from the command line::

    aequitas-report

...or, also from the command line, launch the Web front-end::

    python -m serve

(Note: The above command launches a Web server only intended for development.)

Development
===========

Provision your development environment via the shell script ``develop``::

    ./develop

Common development tasks, such as deploying the webapp, may then be handled via ``manage``::

    manage --help

Containerization
================

To build a Docker container of Aequitas::

    docker build -t aequitas .

...or simply via ``manage``::

    manage container build

The Docker image's container defaults to launching the development Web server, though this can be overridden via the Docker "command" and/or "entrypoint".

To run such a container, supporting the Web server, on-the-fly::

    docker run -p 5000:5000 -e "HOST=0.0.0.0" aequitas

...or, manage a development container via ``manage``::

    manage container [create|start|stop]

Find out more at `the documentation  <https://dssg.github.io/aequitas/>`_.

To contact the team, please email us at [aequitas at uchicago dot edu]


Citing Aequitas
====

If you use Aequitas in a scientific publication, we would appreciate citations to the following paper:

Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld, Loren Hinkson, Jesse London, Rayid Ghani, Aequitas: A Bias and Fairness Audit Toolkit,  arXiv preprint arXiv:1811.05577 (2018). ( `PDF <https://arxiv.org/pdf/1811.05577.pdf>`_)


   @article{2018aequitas,
     title={Aequitas: A Bias and Fairness Audit Toolkit},
     author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid},
     journal={arXiv preprint arXiv:1811.05577},
     year={2018}}
|
|
|
|
|
|


© 2018 Center for Data Science and Public Policy - University of Chicago
