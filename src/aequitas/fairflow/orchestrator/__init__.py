import datetime
import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable, Optional, Union

from hpt import OptunaTuner
from omegaconf import DictConfig, ListConfig
from optuna.samplers import BaseSampler

from ..datasets import Dataset
from ..methods.postprocessing.identity import Identity as PostIdentity
from ..methods.postprocessing.threshold import Threshold
from ..methods.preprocessing.identity import Identity as PreIdentity
from ..optimization import ObjectiveFunction
from ..utils import ConfigReader, create_logger, import_object


class Orchestrator:
    SAMPLERS_MODULE = "optuna.samplers."

    POSSIBLE_ARTIFACTS = (
        "results",
        "methods",
        "predictions",
        "optimizer",
        "transformed_datasets",
    )

    def __init__(
        self,
        config_file: Optional[Path] = None,
        config: Optional[dict] = None,
        default_fields: Iterable[str] = ("methods", "datasets"),
        save_artifacts: bool = True,
        save_folder: Optional[Path] = Path("artifacts"),
        artifacts: Iterable[str] = ("results", "methods", "predictions"),
    ):
        """
        Initialize the Orchestrator of an experiment.

        Parameters
        ----------
        config_file : pathlib.Path
            Path to the configuration file.
        default_fields : Iterable[str], optional
            Default fields to include in the configuration, by default ("methods",
            "datasets").
        save_artifacts : bool, optional
            Whether to save artifacts, by default True.
        save_folder : numpy.ndarray, optional
            Path to the folder where artifacts will be saved, by default None.
        artifacts : numpy.ndarray, optional
            Artifacts to save, by default ("results", "methods", "predictions").
        """
        # Initialize logger
        self.logger = create_logger("Benchmark")
        self.logger.info("Instantiating Benchmark class.")

        # Read config file
        if config is not None:
            self.config = config
        elif config_file is not None:
            self.config_reader = ConfigReader(
                config_file, default_fields=default_fields
            )
            self.config = self.config_reader.config
        else:
            raise ValueError("Missing configuration file")

        # Check if optimization parameters are provided
        if not hasattr(self.config, "optimization"):
            raise ValueError("Provide optimization parameters in the config file.")

        # Debug log the benchmark config
        self.logger.debug(f"Benchmark config: {self.config}")

        # Initialize dataset storage for the experiment
        self.datasets: dict = {}

        # Set artifacts to save
        if save_artifacts:
            self.artifacts = tuple(set(self.POSSIBLE_ARTIFACTS) & set(artifacts))
        else:
            self.artifacts = None

        # Instantiate sampler object
        self.sampler = self._instantiate_sampler()
        self.save_folder = save_folder
        self.hash = None

    def _instantiate_sampler(self) -> BaseSampler:
        self.logger.debug("Instantiating sampling object.")
        sampler_path = self.SAMPLERS_MODULE + self.config.optimization.sampler
        sampler = import_object(sampler_path)(**self.config.optimization.sampler_args)  # type: ignore
        return sampler

    @classmethod
    def read_dataset(config: Union[dict, DictConfig]) -> Dataset:
        """Read a dataset from a configuration object."""
        if isinstance(config, dict):
            config = DictConfig(config)
        dataset_class = import_object(config.classpath)
        dataset_object = dataset_class(**config.args)
        return dataset_object

    def _read_datasets(self):
        self.logger.debug("Reading datasets from configuration.")
        for dataset in self.config.datasets:
            for name, configs in dataset.items():  # This iterates once.
                self.logger.debug(f"Reading '{name}'. Configurations: {configs}.")
                dataset_object = self.read_dataset(configs)
                dataset_object.load_data()
                splits = dataset_object.create_splits()
                self.logger.debug(f"Dataset {name} successfully read.")
                for split, data in splits.items():
                    self.logger.debug(f"Splitting {split} into X, y, and s.")
                    X = data.drop(columns=[configs.label, configs.sensitive_attribute])
                    y = data[configs.label]
                    s = data[configs.sensitive_attribute]
                    splits[split] = {"X": X, "y": y, "s": s}
                self.datasets[name] = splits

    def run(self) -> None:
        self.logger.info("Beginning Benchmark.")
        if self.artifacts:
            self.logger.debug("Generating Hash for provided configuration.")
            self.generate_hash()
            exp_folder: Path = self.save_folder / self.hash
            exp_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving objects to '{exp_folder.resolve()}'.")
        self._read_datasets()
        for dataset_id, (dataset_name, dataset) in enumerate(self.datasets.items()):
            self.logger.info(f"Benchmarking in '{dataset_name}'.")
            if self.artifacts:
                dataset_folder: Path = exp_folder / dataset_name
                dataset_folder.mkdir(parents=True, exist_ok=True)
                self.logger.debug(
                    f"Saving dataset-related objects to '{dataset_folder.resolve()}'."
                )
            for method in self.config.methods:
                for method_name, method_items in method.items():
                    self.logger.info(
                        f"Benchmarking '{method_name}' in '{dataset_folder.resolve()}'."
                    )
                    if self.artifacts:
                        method_folder: Path = dataset_folder / method_name
                        method_folder.mkdir(parents=True, exist_ok=True)
                        self.logger.debug(
                            "Saving dataset&method-related objects to "
                            f"'{method_folder.resolve()}'."
                        )

                    # Instantiate one random search optimizer
                    # The objective function is doing the heavy lifting in HPTs code.

                    # Validate if we received pre-processing techniques.
                    if hasattr(method_items, "preprocessing"):
                        _, pre_configs = list(method_items.preprocessing.items())[0]
                        pre = import_object(pre_configs.classpath)
                    else:
                        pre_configs = DictConfig({"args": None})
                        pre = PreIdentity
                    # Validate in the same way the post-processing technique.
                    if hasattr(method_items, "postprocessing"):
                        _, post_configs = list(method_items.postprocessing.items())[0]
                        post = import_object(post_configs.classpath)
                    else:
                        post_configs = DictConfig({"args": None})
                        post = PostIdentity

                    _, inp_configs = list(method_items.inprocessing.items())[0]
                    inp = import_object(inp_configs.classpath)

                    # Kinda big line to get the threshold parameters...
                    threshold_configs = self.config.datasets[dataset_id][
                        dataset_name
                    ].threshold
                    threshold = Threshold(**threshold_configs)
                    # This is a big instantiation, maybe receive datasets all together?
                    self.logger.debug(
                        f"Creating objective function for "
                        f"'{method_name}' in '{dataset_name}'."
                    )
                    objective = ObjectiveFunction(
                        X_train=dataset["train"]["X"],
                        y_train=dataset["train"]["y"],
                        s_train=dataset["train"]["s"],
                        X_val=dataset["validation"]["X"],
                        y_val=dataset["validation"]["y"],
                        s_val=dataset["validation"]["s"],
                        X_test=dataset["test"]["X"],
                        y_test=dataset["test"]["y"],
                        s_test=dataset["test"]["s"],
                        threshold=threshold,
                        preprocessing_method=pre,
                        preprocessing_configs=pre_configs.args,
                        inprocessing_method=inp,
                        inprocessing_configs=inp_configs.args,
                        postprocessing_method=post,
                        postprocessing_configs=post_configs.args,
                        eval_metric="tpr",  # TODO Change this
                        alpha=None,
                        artifacts=self.artifacts,
                        artifacts_folder=method_folder,
                    )

                    tuner = OptunaTuner(objective, sampler=self.sampler)

                    tuner.optimize(
                        n_trials=self.config.optimization.n_trials,
                        n_jobs=self.config.optimization.n_jobs,
                    )

                    if "results" in self.artifacts:
                        with open(method_folder / "results.pickle", "wb") as file:
                            pickle.dump(objective._models_results, file)

    def generate_hash(self):
        def dt_handler(x):
            if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
                return x.isoformat()
            elif isinstance(x, DictConfig):
                return dict(x)
            elif isinstance(x, ListConfig):
                return list(x)
            raise TypeError("Unknown type")

        self.hash = hashlib.md5(
            json.dumps(self.config, default=dt_handler, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.logger.debug(f"Hash generated: {self.hash}.")
