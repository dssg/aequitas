from typing import Union

import numpy as np
import pandas as pd


def create_bootstrap_estimates(
    results: list[Result],
    configs_per_trial: Union[float, list[int]],
    alpha_definition: Union[float, list[float]],
    evaluate_on: str,
    performance_metric: str,
    fairness_metric: str,
    n_trials: int = 100,
    seed: int = 42,
):
    """Method to create bootstrap estimates for the performance and fairness metrics.

    Works both for the creation of samples over the different alpha values with the same
    sample size and for over different sample sizes for the same alpha value.

    Parameters
    ----------
    results : list[Result]
        List of results objects.
    n_trials : int
        Number of bootstrap samples.
    seed : int
        Seed for the random number generator and sampling.
    configs_per_trial : Union[float, list[int]]
        Number of configurations to sample per trial. If float, it is interpreted as a
        percentage of the total number of configurations. If list, it is interpreted as
        the number of configurations to sample for each trial.
    alpha_definition : Union[float, list[float]]
        Alpha values to use for the bootstrap samples. If float, it is interpreted as a
        single alpha value. If list, it is interpreted as a list of alpha values.
    evaluate_on : str
        Whether to evaluate on the validation or test set.
    performance_metric : str
        Name of the performance metric to use.
    fairness_metric : str
        Name of the fairness metric to use.
    """
    if isinstance(alpha_definition, list) and isinstance(configs_per_trial, list):
        raise ValueError(
            "Only one of alpha_definition and configs_per_trial can be a list"
        )

    # prepare the results object
    if evaluate_on == "validation":
        results_property = "validation_results"
    elif evaluate_on == "test":
        results_property = "test_results"
    else:
        raise ValueError("EVALUATE_ON must be either 'validation' or 'test'")

    # for each result in results, get the results_property
    results_dict = {}
    for index, result in enumerate(results):
        results_dict[index] = getattr(result, results_property)

    index = []
    df_data = {}
    for key, value in results_dict.items():
        index.append(key)
        for k, v in value.items():
            values = df_data.get(k, [])
            values.append(v)
            df_data[k] = values

    models = pd.DataFrame(df_data, index=index)

    # create the alpha weighted column
    if isinstance(alpha_definition, float):
        alphas = [alpha_definition]
    else:
        alphas = alpha_definition

    if isinstance(configs_per_trial, float):
        n_configs = [configs_per_trial]
    else:
        n_configs = configs_per_trial

    for alpha in alphas:
        models[f"alpha_{alpha}"] = models[performance_metric] * alpha + models[
            fairness_metric
        ] * (1 - alpha)

    np.random.seed(seed)
    sampling_seeds = np.random.choice(n_trials * 1000, n_trials, replace=False)

    final_results = {}
    if isinstance(configs_per_trial, float):
        for alpha in alphas:
            final_results[alpha] = {
                "performance": [],
                "fairness": [],
                "alpha_weighted": [],
            }
    else:
        for n in n_configs:
            final_results[n] = {"performance": [], "fairness": [], "alpha_weighted": []}

    # If we are iterating over alphas:
    if isinstance(configs_per_trial, float):
        for index, seed in enumerate(sampling_seeds):
            n_models_to_sample = int(round(configs_per_trial * models.shape[0], 0))
            indexes_to_sample = np.random.choice(
                list(results_dict.keys()), n_models_to_sample, replace=True
            )
            sampled_models = models.loc[indexes_to_sample]
            for alpha in alphas:
                selected_model = sampled_models[
                    sampled_models[f"alpha_{alpha}"]
                    == sampled_models[f"alpha_{alpha}"].max()
                ]
                final_results[alpha]["performance"].append(
                    selected_model[performance_metric].values[0]
                )
                final_results[alpha]["fairness"].append(
                    selected_model[fairness_metric].values[0]
                )
                final_results[alpha]["alpha_weighted"].append(
                    selected_model[f"alpha_{alpha}"].values[0]
                )

    # If we are iterating over trials:
    else:
        for index, seed in enumerate(sampling_seeds):
            for n in n_configs:
                indexes_to_sample = np.random.choice(
                    list(results_dict.keys()), n, replace=True
                )
                sampled_models = models.loc[indexes_to_sample]
                selected_model = sampled_models[
                    sampled_models[f"alpha_{alpha_definition}"]
                    == sampled_models[f"alpha_{alpha_definition}"].max()
                ]
                final_results[n]["performance"].append(
                    selected_model[performance_metric].values[0]
                )
                final_results[n]["fairness"].append(
                    selected_model[fairness_metric].values[0]
                )
                final_results[n]["alpha_weighted"].append(
                    selected_model[f"alpha_{alpha_definition}"].values[0]
                )

    return final_results
