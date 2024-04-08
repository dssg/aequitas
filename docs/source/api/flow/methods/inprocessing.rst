:mod:`inprocessing` classes
---------------------------------------

The :mod:`inprocessing` methods introduce modifications in the algorithms to ensure fairness during model training.
The :class:`InProcessing` class is the abstract class from which all specific preprocessing methods derive.
 
.. curentmodule:: aequitas.flow.methods
.. autoclass:: InProcessing
    :members:


---------------------------------------
Fair GBM
---------------------------------------

The FairGBM model is a novel method of InProcessing, where a boosting trees algorithm (LightGBM) is subject to pre-defined fairness constraints.

.. currentmodule:: aequitas.flow.methods.inprocessing
.. autoclass:: FairGBM
    :members:


---------------------------------------
Fairlearn Classifier
---------------------------------------

:class:`FairlearnClassifier` creates a model from the Fairlearn package.
Especially designed for the ExponentiatedGradient and GridSearch methods.

.. currentmodule:: aequitas.flow.methods.inprocessing
.. autoclass:: FairlearnClassifier
    :members:
