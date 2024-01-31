from typing import Literal, Union

from omegaconf import DictConfig
import pandas as pd

from . import _configs
from .experiment import Experiment


class DefaultExperiment(Experiment):
    def __init__(
        self,
        dataset_config: Union[DictConfig, dict],
        methods: Union[
            list[Literal["preprocessing", "inprocessing"]], Literal["all"]
        ] = "all",
        experiment_size: Literal["test", "small", "medium", "large"] = "small",
        use_baseline: bool = True,
    ):
        """Default experiment configuration.

        Allows for a quick setup of an aequitas flow experiment. The user can choose
        between different experiment sizes, and select which methods to include in the
        experiment.

        Parameters
        ----------
        dataset_config : Union[DictConfig, dict]
            Dataset configuration.
        methods : Union[list[Literal["preprocessing", "inprocessing"]], Literal["all"]], optional
            Methods to include in the experiment. If "all", all methods will be included.
            Defaults to "all".
        experiment_size : Literal["test", "small", "medium", "large"], optional
            Experiment size. Defaults to "small".

        Raises
        ------
        ValueError
            If the methods or experiment size are not valid.
        """
        # Validate methods:
        if methods == "all":
            default_methods = [
                "preprocessing",
                "inprocessing",
            ]
        elif isinstance(methods, list):
            default_methods = methods
        else:
            raise ValueError(
                "Invalid methods value. Try one of "
                f"{['all', ['preprocessing', 'inprocessing']]}."
            )
        method_configs = []
        for method in default_methods:
            if use_baseline:
                method_configs.extend(_configs.baseline_methods)
            if method == "preprocessing":
                method_configs.extend(_configs.preprocessing_methods)
            elif method == "inprocessing":
                method_configs.extend(_configs.inprocessing_methods)

        # Validate experiment size:
        if experiment_size == "test":
            experiment_config = _configs.test_experiment
        elif experiment_size == "small":
            experiment_config = _configs.small_experiment
        elif experiment_size == "medium":
            experiment_config = _configs.medium_experiment
        elif experiment_size == "large":
            experiment_config = _configs.large_experiment
        else:
            raise ValueError(
                "Invalid experiment_size value. Try one of "
                f"{['test', 'small', 'medium', 'large']}."
            )
        # Update experiment config:
        config = {
            "methods": method_configs,
            "datasets": [dataset_config],
            "optimization": experiment_config,
        }
        super().__init__(config=DictConfig(config))

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        target_feature: str,
        sensitive_feature: str,
        other_dataset_args: dict = None,
        threshold_type: str = "fixed",
        score_threshold: float = 0.5,
        dataset_name: str = "Dataset",
        methods: Union[
            list[Literal["preprocessing", "inprocessing"]], Literal["all"]
        ] = "all",
        experiment_size: Literal["test", "small", "medium", "large"] = "small",
        use_baseline: bool = True,
    ):
        """Create a DefaultExperiment from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame with the dataset to be used in the experiment.
        target_feature : str
            Name of the column containing the label.
        sensitive_feature : str
            Name of the column containing the sensitive attribute.
        other_dataset_args : dict, optional
            Other arguments to pass to the dataset. Defaults to None.
        threshold_type : str, optional
            Threshold type. Defaults to "fixed".
        score_threshold : float, optional
            Score threshold. Defaults to 0.5.
        dataset_name : str, optional
            Dataset name. Defaults to "Dataset".
        methods : Union[list[Literal["preprocessing", "inprocessing"]], Literal["all"]], optional
            Methods to include in the experiment. If "all", all methods will be included.
            Defaults to "all".
        experiment_size : Literal["test", "small", "medium", "large"], optional
            Experiment size. Defaults to "small".
        use_baseline : bool, optional
            Whether to include the baseline methods. Defaults to True.

        Returns
        -------
        DefaultExperiment
            Default experiment.

        Raises
        ------
        ValueError
            If the methods or experiment size are not valid.
        """
        dataset_config = {
            dataset_name: {
                "classpath": "aequitas.flow.datasets.GenericDataset",
                "threshold": {
                    "type": threshold_type,
                    "value": score_threshold,
                },
                "args": {
                    "df": df,
                    "target_feature": target_feature,
                    "sensitive_feature": sensitive_feature,
                    **(other_dataset_args or {}),
                }
            }
        }
        return cls(
            dataset_config=dataset_config,
            methods=methods,
            experiment_size=experiment_size,
            use_baseline=use_baseline,
        )
