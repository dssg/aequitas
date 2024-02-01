from typing import Literal, Union

from omegaconf import DictConfig
import pandas as pd

from . import _configs
from .experiment import Experiment


class DefaultExperiment(Experiment):
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        sensitive_column: str,
        categorical_columns: list[str] = [],
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
        """Default experiment configuration.

        Allows for a quick setup of an aequitas flow experiment. The user can choose
        between different experiment sizes, and select which methods to include in the
        experiment.

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame with the dataset to be used in the experiment.
        label_column : str
            Name of the column containing the label.
        sensitive_column : str
            Name of the column containing the sensitive attribute.
        categorical_columns : list[str], optional
            List of categorical columns. Defaults to [].
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
                    "label_column": label_column,
                    "sensitive_column": sensitive_column,
                    "categorical_columns": categorical_columns,
                    **(other_dataset_args or {}),
                },
            }
        }

        config = self._generate_config(
            dataset_config=dataset_config,
            methods=methods,
            experiment_size=experiment_size,
            use_baseline=use_baseline,
        )

        super().__init__(config=config)

    @staticmethod
    def _generate_config(
        dataset_config: dict,
        methods: Union[list[Literal["preprocessing", "inprocessing"]], Literal["all"]],
        experiment_size: Literal["test", "small", "medium", "large"],
        use_baseline: bool,
    ):
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
        # Generate experiment config:
        return {
            "methods": method_configs,
            "datasets": [dataset_config],
            "optimization": experiment_config,
        }

    @classmethod
    def from_config(
        cls,
        dataset_config: Union[DictConfig, dict],
        methods: Union[
            list[Literal["preprocessing", "inprocessing"]], Literal["all"]
        ] = "all",
        experiment_size: Literal["test", "small", "medium", "large"] = "small",
        use_baseline: bool = True,
    ):
        """Create a DefaultExperiment from a pandas DataFrame.

        Parameters
        ----------
        dataset_config : Union[DictConfig, dict]
            Dataset configuration.
        methods : Union[list[Literal["preprocessing", "inprocessing"]], Literal["all"]], optional
            Methods to include in the experiment. If "all", all methods will be included.
            Defaults to "all".
        experiment_size : Literal["test", "small", "medium", "large"], optional
            Experiment size. Defaults to "small".

        Returns
        -------
        DefaultExperiment
            Default experiment.

        Raises
        ------
        ValueError
            If the methods or experiment size are not valid.
        """
        config = cls._generate_config(
            dataset_config=dataset_config,
            methods=methods,
            experiment_size=experiment_size,
            use_baseline=use_baseline,
        )

        return super().__init__(config=config)
