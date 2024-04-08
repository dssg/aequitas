:mod:`Methods` Module
============================

The :mod:`Methods` module comprises various submodules that address different phases of the FairML pipeline.
Below is a breakdown of each submodule, detailing their purpose and functionality.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Preprocessing
-------------
This submodule includes methods that modify the training data with the aim of training fairer models.

.. toctree::
   :maxdepth: 2
   :glob:
   
   methods/preprocessing

Inprocessing
------------
Inprocessing methods introduce modifications in the algorithms to ensure fairness during model training.

.. toctree::
   :maxdepth: 2
   :glob:
   
   methods/inprocessing

Postprocessing
--------------
After obtaining the model predictions for a given task, postprocessing methods can be applied to improve the fairness of the decisions.

.. toctree::
   :maxdepth: 2
   :glob:
   
   methods/postprocessing
