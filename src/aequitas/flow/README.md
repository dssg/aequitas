
<img src="https://github.com/dssg/aequitas/blob/master/src/aequitas_webapp/static/images/aequitas_fairflow_header.png">

# Comparison of Fair ML Methods


Aequitas Fairflow is an open-source project for research data scientists and practitioners to compare different methods of Fair ML and aid in finding the best models for a given dataset with fairness concerns.

\<Add image here\>

# Installation

Aequitas Fairflow is compatible with: **Python 3.7+**

Aequitas Fairflow is bundled with Aequitas from **version 0.43.0 onwards**. Install Aequitas using pip: 

```bash
pip install "aequitas>=0.43.0"
```


# 30 Seconds to Aequitas Fairflow

The main entry point for Fairflow is through the `Orchestrator` class. This allows to run an experiment if the configurations are properly set.

 ```python 
 from aequitas.fairflow import Orchestrator

 # Instantiate the benchmark and define the folder where to save results
 benchmark = Benchmark(config_path, save_folder=Path("results"))
 benchmark.run()
 ```

The definition of configuration files and instantiating an orchestrator will run an experiment with the defined methods, datasets, and optimization configurations.

Aequitas Fairflow has two main customizable components: Datasets and Methods. 
Datasets allow the user to load a predetermined dataset from the literature or from their path. The interface follows a similar pattern between the datasets in the package: `instantiation` -> `load` -> `split`.
```python
from aequitas.fairflow.datasets import GenericDataset

dataset = GenericDataset(path="dataset.csv")
dataset.load_data()
splits = dataset.create_splits()
```
The methods are splitted according to pre-, in- and post-processing. Pre-processing methods transform the dataset; In-processing methods score the dataset; Post-processing methods transform the scores. In the example bellow we can see how each of the methods can be instantiated and used in sequence:

```python
from aequitas.fairflow.methods.preprocessing import PrevalenceSampling
from aequitas.fairflow.methods.inprocessing import FairGBM
from aequitas.fairflow.methods.postprocessing import GroupThreshold

sampling = PrevalenceSampling()
sampling.fit(*splits["train"])
sampled_data = sampling.transform(*splits["train"])

model = FairGBM()
model.fit(*sampled_data)
preds = model.predict_proba(*splits["validation"])

threshold = GroupThreshold()
threshold.fit(*splits["validation"], preds)
final_scores = threshold.transform(*splits["validation"], preds)
```

# Configurations

The configurations follow a structure leveraged by `Hydra`. An example of this structure:
```bash
configs
├── experiment_config.yaml
├── datasets
│   ├── dataset_1.yaml
│   ├── dataset_2.yaml
│   │   ...
│   └── dataset_n.yaml
└── methods
    ├── method_1
    │   ├── inprocessing
    │   │   └── m1_inprocessing.yaml
    │   ├── postprocessing
    │   │   └── m1_postprocessing.yaml
    │   └── preprocessing
    │       └── m1_preprocessing.yaml
    ├── method_1.yaml
    ├── method_2
    │   └── ...
    ├── method_2.yaml
    │   ...
    ├── method_n
    │   └── ...
    └── method_n.yaml
```
You can see in detail how to define each of the configuration files in the \'[examples]()\' directory.

# Abstract
We introduce Aequitas Fairflow, an open-source, comprehensive framework developed for end-to-end experimentation with Fair Machine Learning (Fair ML) methodologies. This  framework offers unified interfaces for a wide range of techniques, metrics, optimization procedures, and datasets, designed with effective default settings. It is designed to the facilitate that activities two types of users typically engage in: 1) researchers who have developed a new bias mitigation method and want to compare its performance against other previous methods, and 2) practitioners who have a specific ML problem at hand and want to decide which bias mitigation method to use. Aequitas Fairflow incorporates an automated pipeline that enables fairness-aware model training, hyperparameter selection, and evaluation. It facilitates seamless experimental runs and offers custom analysis of results tailored to user requirements, from ML researchers to practitioners. In its initial version, the platform supports seven Fair ML techniques, encompassing pre-, in-, and post-processing stages, evaluated on eleven publicly accessible tabular datasets. We expect that Aequitas Fairflow contributes to the systematic, rigorous, reproducible, and transparent evaluation of Fair ML methodologies, paving the way for their widespread real-world adoption and tangible societal impact.

# Citing Aequitas Fairflow

If you use Aequitas Fairflow in a scientific publication, we would appreciate citations to the following papers:

Sérgio Jesus, Pedro Saleiro, Rita P. Ribeiro, João Gama, Pedro Bizarro, Rayid Ghani, Aequitas Fairflow: Towards a Transparent, Reproducible and Rigorous Comparison of Fair ML Methods (2023). ([PDF]())

Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld, Loren Hinkson, Jesse London, Rayid Ghani, Aequitas: A Bias and Fairness Audit Toolkit,  arXiv preprint arXiv:1811.05577 (2018). ([PDF](https://arxiv.org/pdf/1811.05577.pdf))

```bib
@article{2023fairflow,
   title={Aequitas Fairflow: Towards a Transparent, Reproducible and Rigorous Comparison of Fair ML Methods},
   author={Jesus, S{\'{e}}rgio and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~{a}}o and Bizarro, Pedro and Ghani Rayid}, year={2023}}
   
@article{2018aequitas,
    title={Aequitas: A Bias and Fairness Audit Toolkit},
    author={Saleiro, Pedro and Kuester, Benedict and Stevens, Abby and Anisfeld, Ari and Hinkson, Loren and London, Jesse and Ghani, Rayid}, journal={arXiv preprint arXiv:1811.05577}, year={2018}}
``` 