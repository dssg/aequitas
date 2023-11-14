import logging
import sys
from pathlib import Path
from typing import Optional


class Defaults:
    # Default logging values. Can be changed before running the benchmark.
    def __init__(self) -> None:
        self._default_level = logging.INFO
        self._default_logging_format = (
            "[%(levelname)s] %(asctime)s %(name)s - %(message)s"
        )
        self._default_time_format = "%Y-%m-%d %H:%M:%S"
        self._default_file = None

    @property
    def default_level(self) -> int:
        return self._default_level

    @default_level.setter
    def default_level(self, level: int) -> None:
        self._default_level = level

    @property
    def default_logging_format(self) -> str:
        return self._default_logging_format

    @default_logging_format.setter
    def default_logging_format(self, logging_format: str) -> None:
        self._default_logging_format = logging_format

    @property
    def default_time_format(self) -> str:
        return self._default_time_format

    @default_time_format.setter
    def default_time_format(self, time_format: str) -> None:
        self._default_time_format = time_format

    @property
    def default_file(self) -> Optional[Path]:
        return self._default_file

    @default_file.setter
    def default_file(self, file: Optional[Path]) -> None:
        self._default_file = file


def create_logger(
    name: str,
    level: Optional[int] = None,
    format: Optional[str] = None,
    time_format: Optional[str] = None,
    file: Optional[Path] = None,
) -> logging.Logger:
    """Setups a logger and their handlers.

    To change the default values, alter the global variables above.

    Parameters
    ----------
    name : str
        Name for the logger.
    level : int , optional
        Logging level (e.g. debug, info.). If none, default is used.
    format : str, optional
        Format of logging message. If none, default is used.
    time_format : str, optional
        Format of timestamps. If none, default is used (global variable).
    file : Path, optional
        Path to logging file. If none, default is used.
    Returns
    -------
    logging.Logger
        The requested logger.
    """
    # check if logger already exists, return if it does.
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    # logging level
    level = level if level else defaults.default_level

    # logging format
    format = format if format else defaults.default_logging_format
    time_format = time_format if time_format else defaults.default_time_format

    formatter = logging.Formatter(format, time_format)

    # logging file
    file = file if file else defaults.default_file

    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # Add stream handler (stdout) to logger
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    logger.addHandler(stdout_handler)

    # add file handler if requested
    if file:
        file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def clean_handlers() -> None:
    """
    Cleans the handlers of the root logger.

    Used in Google Colab, as it has one default handler set, and propagates logs from
    our package.
    """
    root = logging.getLogger()
    root.handlers = []


defaults = Defaults()
