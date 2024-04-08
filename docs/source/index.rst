.. aequitas documentation master file, created by
   sphinx-quickstart on Fri Apr 13 15:28:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Aequitas: Bias Auditing & "Correction" Toolkit
==============================================


`aequitas` is an open-source bias auditing and Fair ML toolkit for data scientists, machine learning researchers, and policymakers. We provide an easy-to-use and transparent tool for auditing predictors of ML models, as well as experimenting with "correcting biased model" using Fair ML methods in binary classification settings.

For more context around dealing with bias and fairness issues in AI//ML systems, take a look at our `detailed tutorial <https://dssg.github.io/fairness_tutorial/>`_ and related publications.

.. note::

   **Version 1.0.0: Aequitas Flow - Optimizing Fairness in ML Pipelines**

   Explore Aequitas Flow, our latest update in version 1.0.0, designed to augment bias audits with bias mitigation and allow enrich  experimentation with Fair ML methods using our new, streamlined capabilities. 
      
Features of the Toolkit
-----------------------

- **Metrics**: Audits based on confusion matrix-based metrics with flexibility to select the more important ones depending on use-case.
- **Plotting options**: The major outcomes of bias auditing and experimenting offer also plots adequate to different user objectives. 
- **Fair ML methods**: Interface and implementation of several Fair ML methods, including pre-, in-, and post-processing methods.
- **Datasets**: Two "families" of datasets included, named `BankAccountFraud <https://arxiv.org/pdf/2211.13358>`_ and `FolkTables <https://arxiv.org/abs/2108.04884>`_.
- **Extensibility**: Adapted to receive user-implemented methods, with intuitive interfaces and method signatures.
- **Reproducibility**: Option to save artifacts of Experiments, from the transformed data to the fitted models and predictions.
- **Modularity**: Fair ML Methods and default datasets can be used individually or integrated in an `Experiment`.
- **Hyperparameter optimization**: Out of the box integration and abstraction of `Optuna <https://github.com/optuna/optuna>`_'s hyperparameter optimization capabilities for experimentation.

Fair ML Methods
---------------


We support a range of methods designed to address bias and discrimination in different stages of the ML pipeline.

.. list-table:: Preprocessing methods
   :widths: 25 25 
   :header-rows: 1

   * - Method
     - Description
   * - Data Repairer
     - Transforms the data distribution so that a given feature distribution is marginally independent of the sensitive attribute, s.
   * - Label Flipping
     - Flips the labels of a fraction of the training data according to the Fair Ordering-Based Noise Correction method.
   * - Prevalence Sampling
     - Generates a training sample with controllable balanced prevalence for the groups in dataset, either by undersampling or oversampling.
   * - Massaging
     - Flips selected labels to reduce prevalence disparity between groups.
   * - Correlation Suppression
     - Removes features that are highly correlated with the sensitive attribute.
   * - Feature Importance Suppression
     - Iterively removes the most important features with respect to the sensitive attribute.
   

.. list-table:: Inprocessing methods
   :widths: 25 25 
   :header-rows: 1

   * - Method
     - Description
   * - FairGBM
     - Novel method where a boosting trees algorithm (LightGBM) is subject to pre-defined fairness constraints.
   * - Fairlearn Classifier
     - Models from the Fairlearn reductions package. Possible parameterization for ExponentiatedGradient and GridSearch methods.


.. list-table:: Postprocessing methods
   :widths: 25 25 
   :header-rows: 1

   * - Method
     - Description
   * - Group Threshold
     - Adjusts the threshold per group to obtain a certain fairness criterion (e.g., all groups with 10% FPR)
   * - Balanced Group Threshold
     - Adjusts the threshold per group to obtain a certain fairness criterion, while satisfying a global constraint (e.g., Demographic Parity with a global FPR of 10%)
   
Explore the documentation
-------------------------

.. toctree::
   :maxdepth: 1
   :glob:
   
   installation
   user_guide
   api/index
   examples/index

Citing Aequitas
---------------

If you use Aequitas in a scientific publication, we would appreciate citations to the following paper:

Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld, Loren Hinkson, Jesse London, Rayid Ghani, Aequitas: A Bias and Fairness Audit Toolkit,  arXiv preprint arXiv:1811.05577 (2018). `PDF <https://arxiv.org/pdf/1811.05577.pdf>`_

.. code-block:: bib

   @article{2018aequitas,
     title={Aequitas: A Bias and Fairness Audit Toolkit},
     author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid}, 
     journal={arXiv preprint arXiv:1811.05577}, 
     year={2018}}


