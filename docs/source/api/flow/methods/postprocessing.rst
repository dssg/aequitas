
:mod:`postprocessing` classes
---------------------------------------

After obtaining the model predictions for a given task, :mod:`postprocessing` methods can be applied to improve the fairness of the decisions.
The :class:`PostProcessing` class is the abstract class from which all specific preprocessing methods derive.
 
.. curentmodule:: aequitas.flow.methods
.. autoclass:: PostProcessing
    :members:


---------------------------------------
Thresholding
---------------------------------------

The :mod:`Threshold` class applies thresholding to the model's predictions.

.. currentmodule:: aequitas.flow.methods.postprocessing
.. autoclass:: Threshold
    :members:

The :mod:`GroupThreshold` class adjusts the prediction scores based on a threshold for multiple groups in the dataset.

.. currentmodule:: aequitas.flow.methods.postprocessing
.. autoclass:: GroupThreshold
    :members:

The :mod:`BalancedGroupThreshold` class adjusts the prediction scores based on a balanced threshold for multiple groups in the dataset.

.. currentmodule:: aequitas.flow.methods.postprocessing
.. autoclass:: BalancedGroupThreshold
    :members:
