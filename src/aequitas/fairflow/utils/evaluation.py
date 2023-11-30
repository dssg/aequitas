from typing import Literal, Union

import numpy as np
import pandas as pd

from ..evaluation import Result


# Define constants
SEED_MULTIPLIER = 1000


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
    alpha_points: Union[float, list[float]],
    performance_metric: str,
    fairness_metric: str,
) -> pd.DataFrame:
    """Method to calculate the alpha weighted metric for the given models.

    Parameters
    ----------
    models : pd.DataFrame
        DataFrame with the results for the given dataset split.
    alpha_points : Union[float, list[float]]
        Alpha values to use for the bootstrap samples. If float, it is interpreted as a
        single alpha value. If list, it is interpreted as a list of alpha values.
    performance_metric : str
        Name of the performance metric to use.
    fairness_metric : str
        Name of the fairness metric to use.
    """

    def calculate_alpha_weighted_score(row, alpha, performance_metric, fairness_metric):
        return row[performance_metric] * alpha + row[fairness_metric] * (1 - alpha)

    if isinstance(alpha_points, float):
        alpha_points = [alpha_points]

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


def _create_final_results(keys):
    """Method to create the final results object.

    Parameters
    ----------
    keys : list
        List of keys to use for the final results object.
    """
    final_results = {}
    for key in keys:
        final_results[key] = {"performance": [], "fairness": [], "alpha_weighted": []}
    return final_results


def _sample_models(models, n_models_to_sample, seed):
    """Method to sample models from the given DataFrame.

    Parameters
    ----------
    models : pd.DataFrame
        DataFrame with the results for the given dataset split.
    n_models_to_sample : int
        Number of models to sample.
    seed : int
        Seed for the random number generator and sampling.
    """
    np.random.seed(seed)
    indexes_to_sample = np.random.choice(
        list(models.index),
        n_models_to_sample,
        replace=True,
    )
    sampled_models = models.loc[list(set(indexes_to_sample))]
    return sampled_models


def _sample_models_bootstrap(models, n_models_to_sample, seed):
    """Method to sample models from the given DataFrame.

    Parameters
    ----------
    models : pd.DataFrame
        DataFrame with the results for the given dataset split.
    n_models_to_sample : int
        Number of models to sample.
    seed : int
        Seed for the random number generator and sampling.
    """
    np.random.seed(seed)
    indexes_to_sample = np.random.choice(
        list(models.index),
        n_models_to_sample,
        replace=True,
    )

    sampled_models = {}
    for i in range(n_models_to_sample):
        sampled_models[n_models_to_sample-i] = models.loc[list(set(indexes_to_sample))]
        indexes_to_sample = indexes_to_sample[:-1]
    return sampled_models


def _get_max(models, alpha, best_models):
    """Method to get the model with the highest alpha weighted metric.

    Parameters
    ----------
    models : pd.DataFrame
        DataFrame with the results for the given dataset split.
    alpha : float
        Alpha value to use for the bootstrap samples.
    best_models : dict
        Dictionary with the indexes of the best models for each alpha value.
    """
    max_index = next(idx for idx in best_models[alpha] if idx in models.index)
    return_val = models.loc[max_index]
    return return_val


def _get_best_models_ordered(models, alpha_points):
    if isinstance(alpha_points, float):
        alpha_points = [alpha_points]
    best_models_ordered = {}
    for alpha in alpha_points:
        best_models_ordered[alpha] = (
            models[f"alpha_{alpha}"].sort_values(ascending=False).index
        )
    return best_models_ordered


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

    is_alpha_bootstrap = isinstance(bootstrap_size, float)

    # prepare the results object
    models = _prepare_results(results, evaluate_on)

    # Add alpha metrics to the metrics DataFrame
    models = _calculate_alpha_weighted_metric(
        models,
        alpha_points,
        performance_metric,
        fairness_metric,
    )

    best_models = _get_best_models_ordered(models, alpha_points)

    np.random.seed(seed)
    sampling_seeds = np.random.choice(
        n_trials * SEED_MULTIPLIER, n_trials, replace=False
    )

    if is_alpha_bootstrap:
        final_results = _create_final_results(alpha_points)
        n_models_to_sample = int(round(bootstrap_size * models.shape[0], 0))
        # Sampling at least one model
        if n_models_to_sample == 0:
            n_models_to_sample = 1

    else:
        final_results = _create_final_results(bootstrap_size)

    # If we are iterating over alphas:
    if is_alpha_bootstrap:
        for seed in sampling_seeds:
            sampled_models = _sample_models(models, n_models_to_sample, seed)
            for alpha in alpha_points:
                selected_model = _get_max(sampled_models, alpha, best_models)
                perf = selected_model[performance_metric]
                final_results[alpha]["performance"].append(
                    selected_model[performance_metric]
                )
                final_results[alpha]["fairness"].append(selected_model[fairness_metric])
                final_results[alpha]["alpha_weighted"].append(
                    selected_model[f"alpha_{alpha}"]
                )
                if not isinstance(perf, float):
                    raise ValueError()

    # If we are iterating over bootstraps:
    else:
        for seed in sampling_seeds:
            sampled_models = _sample_models_bootstrap(models, models.shape[0], seed)
            for n in bootstrap_size:
                n_models_to_sample = int(round(n * models.shape[0], 0))
                if n_models_to_sample == 0:
                    n_models_to_sample = 1
                selected_model = _get_max(
                    sampled_models[n_models_to_sample], alpha_points, best_models
                )
                final_results[n]["performance"].append(
                    selected_model[performance_metric]
                )
                final_results[n]["fairness"].append(selected_model[fairness_metric])
                final_results[n]["alpha_weighted"].append(
                    selected_model[f"alpha_{alpha_points}"]
                )
    return final_results
