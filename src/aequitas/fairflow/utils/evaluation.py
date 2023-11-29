from typing import Literal, Union

import numpy as np
import pandas as pd

from ..evaluation import Result


def _prepare_results(
    results: list[Result],
    dataset_split: Literal["train", "validation", "test"],
) -> pd.DataFrame:
    """Method to prepare the results object for the creation of bootstrap estimates.

    Parameters
    ----------
    results : list[Result]
        List of results objects.
    dataset_split : Literal["train", "validation", "test"]
        Dataset split to use for the bootstrap estimates.

    Returns
    -------
    pd.DataFrame
        DataFrame with the results for the given dataset split.
    """
    # Validate the dataset_split argument
    if dataset_split not in ["train", "validation", "test"]:
        raise AttributeError("Invalid split definition for prepare_results")

    results_property = f"{dataset_split}_results"
    metrics = getattr(results[0], results_property).keys()
    data = {
        metric: [getattr(result, results_property)[metric] for result in results]
        for metric in metrics
    }
    return pd.DataFrame(data)


def _calculate_alpha_weighted_metric(
    models: pd.DataFrame,
    alpha_points: list[float],
    performance_metric: str,
    fairness_metric: str,
) -> pd.DataFrame:
    def calculate_alpha_weighted_score(row, alpha, performance_metric, fairness_metric):
        return row[performance_metric] * alpha + row[fairness_metric] * (1 - alpha)

    models = models.copy()
    for alpha in alpha_points:
        models[f"alpha_{alpha}"] = models.apply(
            calculate_alpha_weighted_score,
            axis=1,
            alpha=alpha,
            performance_metric=performance_metric,
            fairness_metric=fairness_metric,
        )
        models = models.copy()
    return models.copy()


def bootstrap_hyperparameters(
    results: list[Result],
    bootstrap_size: Union[float, list[float]],
    alpha_points: Union[float, list[float]],
    evaluate_on: Literal["train", "validation", "test"],
    performance_metric: str,
    fairness_metric: str,
    n_trials: int = 100,
    seed: int = 42,
):
    """Method to create bootstrap estimates for the performance and fairness metrics on
    a random search trial.

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
    bootstrap_size : Union[float, list[float]]
        Number of configurations to sample per trial. If float, it is interpreted as a
        percentage of the total number of configurations. If list, it is interpreted as
        the percentage of configurations to sample for each trial.
    alpha_points : Union[float, list[float]]
        Alpha values to use for the bootstrap samples. If float, it is interpreted as a
        single alpha value. If list, it is interpreted as a list of alpha values.
    evaluate_on : str
        Whether to evaluate on the validation or test set.
    performance_metric : str
        Name of the performance metric to use.
    fairness_metric : str
        Name of the fairness metric to use.
    """
    # Check if only one of alpha_points and bootstrap_size is a list
    # Note: This can be generalizable so both are lists, but it is not needed for now.
    if isinstance(alpha_points, list) ^ isinstance(bootstrap_size, list):
        raise ValueError("Only one of alpha_points and bootstrap_size can be a list")

    # prepare the results object
    models = _prepare_results(results, evaluate_on)

    # Create a list to iterate over the alphas
    alphas = [alpha_points] if isinstance(alpha_points, float) else alpha_points

    samples = [bootstrap_size] if isinstance(bootstrap_size, float) else bootstrap_size

    # Add alpha metrics to the metrics DataFrame
    models = _calculate_alpha_weighted_metric(
        models,
        alphas,
        performance_metric,
        fairness_metric,
    )
    np.random.seed(seed)
    sampling_seeds = np.random.choice(n_trials * 1000, n_trials, replace=False)

    final_results = {}
    if isinstance(bootstrap_size, float):
        for alpha in alphas:
            final_results[alpha] = {
                "performance": [],
                "fairness": [],
                "alpha_weighted": [],
            }
    else:
        for n in samples:
            final_results[n] = {"performance": [], "fairness": [], "alpha_weighted": []}

    # If we are iterating over alphas:
    if isinstance(bootstrap_size, float):
        n_models_to_sample = int(round(bootstrap_size * models.shape[0], 0))

        for seed in sampling_seeds:
            indexes_to_sample = np.random.choice(
                list(models.index), n_models_to_sample, replace=True
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

    # If we are iterating over bootstraps:
    else:
        for _, seed in enumerate(sampling_seeds):
            for n in bootstrap_size:
                n_models_to_sample = int(round(n * models.shape[0], 0))
                indexes_to_sample = np.random.choice(
                    list(models.index), n_models_to_sample, replace=True
                )
                sampled_models = models.loc[indexes_to_sample]
                selected_model = sampled_models[
                    sampled_models[f"alpha_{alpha_points}"]
                    == sampled_models[f"alpha_{alpha_points}"].max()
                ]
                final_results[n]["performance"].append(
                    selected_model[performance_metric].values[0]
                )
                final_results[n]["fairness"].append(
                    selected_model[fairness_metric].values[0]
                )
                final_results[n]["alpha_weighted"].append(
                    selected_model[f"alpha_{alpha_points}"].values[0]
                )

    return final_results
