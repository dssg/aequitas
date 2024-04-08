:mod:`preprocessing` classes
---------------------------------------

The :mod:`preprocessing` methods modify the training data with the aim of training fairer models.
The :class:`PreProcessing` class is the abstract class from which all specific preprocessing methods derive.
 
.. currentmodule:: aequitas.flow.methods
.. autoclass:: PreProcessing
    :members:

---------------------------------------
Data Sampling
---------------------------------------

The :class:`PrevalenceSampling` class generates training sample with balanced prevalence for the groups in dataset.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: PrevalenceSampling
    :members:


---------------------------------------
Distribution Repairer
---------------------------------------

The :class:`DataRepairer` class transforms the data distribution so that a given feature distribution is more or less independent of the sensitive attribute `s`.
This is achieved by matching the conditional distribution `P(X|s)` to the global variable distribution `P(X)`, matching the values of quantiles.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: DataRepairer
    :members:

---------------------------------------
Label Correction
---------------------------------------

The :class:`LabelFlipping` class flips the labels of a fraction of the training data according to the Fair Ordering-Based Noise Correction (Fair-OBNC) method.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: LabelFlipping
    :members:

The :class:`Massaging` class flips selected labels to reduce disparity between groups.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: Massaging
    :members:

---------------------------------------
Suppression
---------------------------------------

The :class:`CorrelationSuppression` class removes features that are highly correlated with the sensitive attribute.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: CorrelationSuppression
    :members:

The :class:`FeatureImportanceSuppression` class iterively removes the most important features with respect to the sensitive attribute.

.. currentmodule:: aequitas.flow.methods.preprocessing
.. autoclass:: FeatureImportanceSuppression
    :members:
