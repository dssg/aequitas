import os
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


class ConfigReader:
    def __init__(self, config_file: Path, default_fields: tuple[str] = ()):
        """Configuration Reader. Reads configuration file in `config_file`.

        If some default field has multiple configurations to be loaded independently,
        use default_fields. Default fields are expected to be in same configuration as
        Hydra. This handles default fields with lists.

        For an example of `default_fields` usage, see the configuration for Benchmarks.

        Parameters
        ----------
        config_file : Path
            Path to the configuration file.
        default_fields : Iterable[str], optional
            Fields with configurations associated, with lists.
        """
        self.config_file = config_file
        config = self.read_config(config_file)
        self.default_fields = default_fields
        # For now, there is no need to create recursive methods here.
        for field in self.default_fields:
            values = config[field]
            field_configs = []
            for value in values:
                value_config_file = config_file.parent / field / value
                value_config = self.read_config(value_config_file)
                field_configs.append(value_config)
            setattr(config, field, field_configs)
        self.config = config

    @staticmethod
    def read_config(config_file: Path) -> DictConfig:
        """Reads given configuration file.

        Absolute path is converted to relative for Hydra methods.

        Parameters
        ----------
        config_file : Path
            Path to the configuration file.

        Returns
        -------
        DictConfig
            Configuration file in a OmegaConf.DictConf format.
        """
        # Create a configuration path which is relative to this file. Hydra works with
        # the directory of the file calling it, not the working directory.
        config_file = Path(os.path.relpath(config_file, Path(__file__).parent))
        # Clear any previous global Hydra configurations.
        GlobalHydra.instance().clear()
        # Split config file path into directory and file name.
        config_path = str(config_file.parent)
        config_name = config_file.name.split(".")[0]
        with hydra.initialize(config_path=config_path, version_base="1.3"):
            config = hydra.compose(config_name=config_name)
        return config
