# *Aequitas*: Bias Auditing & "Correction" Toolkit

[![](https://pepy.tech/badge/aequitas)](https://pypi.org/project/aequitas/)
[![License: MIT](https://badgen.net/pypi/license/aequitas)](https://github.com/dssg/aequitas/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

[comment]: <> (Add badges for coverage when we have tests, update repo for other types of badges!)

`aequitas` is an open-source bias auditing and Fair ML toolkit for data scientists, machine learning researchers, and policymakers. We provide an easy-to-use and transparent tool for auditing predictors of ML models, as well as experimenting with "correcting biased model" using Fair ML methods in binary classification settings.

For more context around dealing with bias and fairness issues in AI//ML systems, take a look at our [detailed tutorial](https://dssg.github.io/fairness_tutorial/) and related publications.


> 
> **Version 1.0.0: Aequitas Flow - Optimizing Fairness in ML Pipelines**
> 
> Explore Aequitas Flow, our latest update in version 1.0.0, designed to augment bias audits with bias mitigation and allow enrich  experimentation with Fair ML methods using our new, streamlined capabilities. 
> 


<p align="center">
  <img src="https://raw.githubusercontent.com/dssg/aequitas/master/docs/_images/aequitas_logo.svg" width="450">
</p>



<p float="left" align="center">
  <a href="#example-notebooks"><img src="https://raw.githubusercontent.com/dssg/aequitas/master/docs/_images/diagram.svg" width="600"/></a>
</p>

## 📥 Installation

```cmd
pip install aequitas
```

or

```cmd
pip install git+https://github.com/dssg/aequitas.git
```

### 📔Example Notebooks supporting various tasks and workflows

| Notebook | Description |
|-|-|
| [Audit a Model's Predictions](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/compas_demo.ipynb) | Check how to do an in-depth bias audit with the COMPAS example notebook or use your own data. |
| [Correct a Model's Predictions](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/aequitas_flow_model_audit_and_correct.ipynb) | Create a dataframe to audit a specific model, and correct the predictions with group-specific thresholds in the Model correction notebook. |
| [Train a Model with Fairness Considerations](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/aequitas_flow_experiment.ipynb) | Experiment with your own dataset or methods and check the results of a Fair ML experiment. |
| [Add your method to Aequitas Flow](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/aequitas_flow_add_method.ipynb) | Learn how to add your own method to the Aequitas Flow toolkit. |


### 🔍 Quickstart on Bias Auditing

To perform a bias audit, you need a pandas `DataFrame` with the following format:

|     | label | score | sens_attr_1 | sens_attr_2 | ... | sens_attr_N |
|-----|-------|-------|-------------|-------------|-----|-------------|
| 0   | 0     | 0     | A           | F           |     | Y           |
| 1   | 0     | 1     | C           | F           |     | N           |
| 2   | 1     | 1     | B           | T           |     | N           |
| ... |       |       |             |             |     |             |
| N   | 1     | 0     | E           | T           |     | Y           |

where `label` is the target variable for your prediction task and `score` is the model output.
Only one sensitive attribute is required; all must be in `Categorical` format.

```python
from aequitas import Audit

audit = Audit(df)
```

To obtain a summary of the bias audit, run:
```python
# Select the fairness metric of interest for your dataset
audit.summary_plot(["tpr", "fpr", "pprev"])
```
<img src="https://raw.githubusercontent.com/dssg/aequitas/master/docs/_images/summary_chart.svg" width="900">

We can also observe a single metric and sensitive attribute:
```python
audit.disparity_plot(attribute="sens_attr_2", metrics=["fpr"])
```
<img src="https://raw.githubusercontent.com/dssg/aequitas/master/docs/_images/disparity_chart.svg" width="900">

### 🧪 Quickstart on experimenting with Bias Reduction (Fair ML) methods

To perform an experiment, a dataset is required. It must have a label column, a sensitive attribute column, and features.  

```python
from aequitas.flow import DefaultExperiment

experiment = DefaultExperiment.from_pandas(dataset, target_feature="label", sensitive_feature="attr", experiment_size="small")
experiment.run()

experiment.plot_pareto()
```

<img src="https://raw.githubusercontent.com/dssg/aequitas/master/docs/_images/pareto_example.png" width="600">

The [`DefaultExperiment`](https://github.com/dssg/aequitas/blob/readme-feedback-changes/src/aequitas/flow/experiment/default.py#L9) class allows for an easier entry-point to experiments in the package. This class has two main parameters to configure the experiment: `experiment_size` and `methods`. The former defines the size of the experiment, which can be either `test` (1 model per method), `small` (10 models per method), `medium` (50 models per method), or `large` (100 models per method). The latter defines the methods to be used in the experiment, which can be either `all` or a subset, namely `preprocessing` or `inprocessing`.

Several aspects of an experiment (*e.g.*, algorithms, number of runs, dataset splitting) can be configured individually in more granular detail in the [`Experiment`](https://github.com/dssg/aequitas/blob/readme-feedback-changes/src/aequitas/flow/experiment/experiment.py#L23) class.


[comment]: <> (Make default experiment this easy to run)

### 🧠 Quickstart on Method Training

Assuming an `aequitas.flow.Dataset`, it is possible to train methods and use their functionality depending on the type of algorithm (pre-, in-, or post-processing).

For pre-processing methods:
```python
from aequitas.flow.methods.preprocessing import PrevalenceSampling

sampler = PrevalenceSampling()
sampler.fit(dataset.train.X, dataset.train.y, dataset.train.s)
X_sample, y_sample, s_sample = sampler.transform(dataset.train.X, dataset.train.y, dataset.train.s)
```

for in-processing methods:
```python
from aequitas.flow.methods.inprocessing import FairGBM

model = FairGBM()
model.fit(X_sample, y_sample, s_sample)
scores_val = model.predict_proba(dataset.validation.X, dataset.validation.y, dataset.validation.s)
scores_test = model.predict_proba(dataset.test.X, dataset.test.y, dataset.test.s)
```

for post-processing methods:
```python
from aequitas.flow.methods.postprocessing import BalancedGroupThreshold

threshold = BalancedGroupThreshold("top_pct", 0.1, "fpr")
threshold.fit(dataset.validation.X, scores_val, dataset.validation.y, dataset.validation.s)
corrected_scores = threshold.transform(dataset.test.X, scores_test, dataset.test.s)
```

With this sequence, we would sample a dataset, train a FairGBM model, and then adjust the scores to have equal FPR per group (achieving Predictive Equality).

## 📜 Features of the Toolkit
- **Metrics**: Audits based on confusion matrix-based metrics with flexibility to select the more important ones depending on use-case.
- **Plotting options**: The major outcomes of bias auditing and experimenting offer also plots adequate to different user objectives. 
- **Fair ML methods**: Interface and implementation of several Fair ML methods, including pre-, in-, and post-processing methods.
- **Datasets**: Two "families" of datasets included, named [BankAccountFraud](https://arxiv.org/pdf/2211.13358) and [FolkTables](https://arxiv.org/abs/2108.04884).
- **Extensibility**: Adapted to receive user-implemented methods, with intuitive interfaces and method signatures.
- **Reproducibility**: Option to save artifacts of Experiments, from the transformed data to the fitted models and predictions.
- **Modularity**: Fair ML Methods and default datasets can be used individually or integrated in an `Experiment`.
- **Hyperparameter optimization**: Out of the box integration and abstraction of [Optuna](https://github.com/optuna/optuna)'s hyperparameter optimization capabilities for experimentation.

### Fair ML Methods

We support a range of methods designed to address bias and discrimination in different stages of the ML pipeline.

<table>
  <tr>
    <th rowspan="2"> Type </th>
    <th rowspan="2"> Method </th>
    <th rowspan="2"> Description </th>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="12"> Pre-processing </td>
    <td rowspan="2"> <a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/data_repairer.py"> Data Repairer </a> </td>
    <td rowspan="2"> Transforms the data distribution so that a given feature distribution is marginally independent of the sensitive attribute, s. </td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"> <a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/label_flipping.py"> Label Flipping </a> </td> 
    <td rowspan="2"> Flips the labels of a fraction of the training data according to the Fair Ordering-Based Noise Correction method. </td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"> <a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/prevalence_sample.py"> Prevalence Sampling </a> </td>
    <td rowspan="2"> Generates a training sample with controllable balanced prevalence for the groups in dataset, either by undersampling or oversampling. </td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/massaging.py">Massaging</td>
    <td rowspan="2">Flips selected labels to reduce prevalence disparity between groups.</td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/correlation_suppression.py">Correlation Suppression</td>
    <td rowspan="2">Removes features that are highly correlated with the sensitive attribute.</td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/feature_importance_suppression.py">Feature Importance Suppression</td>
    <td rowspan="2">Iterively removes the most important features with respect to the sensitive attribute.
    </td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="4"> In-processing </td>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/inprocessing/fairgbm.py"> FairGBM </a> </td>
    <td rowspan="2"> Novel method where a boosting trees algorithm (LightGBM) is subject to pre-defined fairness constraints. </td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/inprocessing/fairlearn_classifier.py">Fairlearn Classifier</td>
    <td rowspan="2"> Models from the Fairlearn reductions package. Possible parameterization for ExponentiatedGradient and GridSearch methods.</td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="4">Post-processing</td>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/postprocessing/group_threshold.py">Group Threshold</td>
    <td rowspan="2">Adjusts the threshold per group to obtain a certain fairness criterion (e.g., all groups with 10% FPR)</td>
  </tr>
  <tr></tr>
  <tr>
    <td rowspan="2"><a href="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/postprocessing/balanced_group_threshold.py">Balanced Group Threshold</td>
    <td rowspan="2">Adjusts the threshold per group to obtain a certain fairness criterion, while satisfying a global constraint (e.g., Demographic Parity with a global FPR of 10%)</td>
  </tr>
  <tr></tr>
</table>

### Fairness Metrics

`aequitas` provides the value of confusion matrix metrics for each possible value of the sensitive attribute columns To calculate fairness metrics. The cells of the confusion metrics are:

| Cell               | Symbol  | Description                                                    | 
|--------------------|:-------:|----------------------------------------------------------------|
| **False Positive** | $FP_g$  | The number of entities of the group with $\hat{Y}=1$ and $Y=0$ |
| **False Negative** | $FN_g$  | The number of entities of the group with $\hat{Y}=0$ and $Y=1$ |
| **True Positive**  | $TP_g$  | The number of entities of the group with $\hat{Y}=1$ and $Y=1$ |
| **True Negative**  | $TN_g$  | The number of entities of the group with $\hat{Y}=0$ and $Y=0$ |

From these, we calculate several metrics:

| Metric                        | Formula                                             | Description                                                                               | 
|-------------------------------|:---------------------------------------------------:|-------------------------------------------------------------------------------------------| 
| **Accuracy**                  | $Acc_g = \cfrac{TP_g + TN_g}{\|g\|}$                | The fraction of correctly predicted entities withing the group.                           |
| **True Positive Rate**        | $TPR_g = \cfrac{TP_g}{TP_g + FN_g}$                 | The fraction of true positives within the label positive entities of a group.             |
| **True Negative Rate**        | $TNR_g = \cfrac{TN_g}{TN_g + FP_g}$                 | The fraction of true negatives within the label negative entities of a group.             |
| **False Negative Rate**       | $FNR_g = \cfrac{FN_g}{TP_g + FN_g}$                 | The fraction of false negatives within the label positive entities of a group.            |
| **False Positive Rate**       | $FPR_g = \cfrac{FP_g}{TN_g + FP_g}$                 | The fraction of false positives within the label negative entities of a group.            |
| **Precision**                 | $Precision_g = \cfrac{TP_g}{TP_g + FP_g}$           | The fraction of true positives within the predicted positive entities of a group.         |
| **Negative Predictive Value** | $NPV_g = \cfrac{TN_g}{TN_g + FN_g}$                 | The fraction of true negatives within the predicted negative entities of a group.         | 
| **False Discovery Rate**      | $FDR_g = \cfrac{FP_g}{TP_g + FP_g}$                 | The fraction of false positives within the predicted positive entities of a group.        |
| **False Omission Rate**       | $FOR_g = \cfrac{FN_g}{TN_g + FN_g}$                 | The fraction of false negatives within the predicted negative entities of a group.        |
| **Predicted Positive**        | $PP_g = TP_g + FP_g$                                |  The number of entities within a group where the decision is positive, i.e., $\hat{Y}=1$. |
| **Total Predictive Positive** | $K = \sum PP_{g(a_i)}$                              | The total number of entities predicted positive across groups defined by $A$              | 
| **Predicted Negative**        | $PN_g = TN_g + FN_g$                                | The number of entities within a group where the decision is negative, i.e., $\hat{Y}=0$   | 
| **Predicted Prevalence**      | $Pprev_g=\cfrac{PP_g}{\|g\|}=P(\hat{Y}=1 \| A=a_i)$ | The fraction of entities within a group which were predicted as positive.                 | 
| **Predicted Positive Rate**   | $PPR_g = \cfrac{PP_g}{K} = P(A=A_i \| \hat{Y}=1)$   | The fraction of the entities predicted as positive that belong to a certain group.        | 

These are implemented in the [`Group`](https://github.com/dssg/aequitas/blob/master/src/aequitas/group.py) class. With the [`Bias`](https://github.com/dssg/aequitas/blob/master/src/aequitas/bias.py) class, several fairness metrics can be derived by different combinations of ratios of these metrics.


## Further documentation

You can find the toolkit documentation [here](https://dssg.github.io/aequitas/).

For more examples of the python library and a deep dive into concepts of fairness in ML, see our [Tutorial](https://github.com/dssg/fairness_tutorial) presented on KDD and AAAI. Visit also the [Aequitas project website](http://dsapp.uchicago.edu/aequitas/).

## Citing Aequitas

To cite Aequitas, please refer to the following papers:

1. Aequitas Flow: Streamlining Fair ML Experimentation (2024) [PDF](https://arxiv.org/pdf/2405.05809)
   
```bib
@article{jesus2024aequitas,
  title={Aequitas Flow: Streamlining Fair ML Experimentation},
  author={Jesus, S{\'e}rgio and Saleiro, Pedro and Jorge, Beatriz M and Ribeiro, Rita P and Gama, Jo{\~a}o and Bizarro, Pedro and Ghani, Rayid and others},
  journal={arXiv preprint arXiv:2405.05809},
  year={2024}
}

```

2. Aequitas: A Bias and Fairness Audit Toolkit (2018) [PDF](https://arxiv.org/pdf/1811.05577.pdf)

```bib
   @article{2018aequitas,
     title={Aequitas: A Bias and Fairness Audit Toolkit},
     author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid}, journal={arXiv preprint arXiv:1811.05577}, year={2018}}
``` 

[Back to top](#aequitas-bias-auditing--fair-ml-toolkit)
