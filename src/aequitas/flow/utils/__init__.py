from typing import TYPE_CHECKING

from .imports import import_object
from .labeled_frame import LabeledFrame
from .logging import create_logger
from .results import read_results, restructure_results

if TYPE_CHECKING:
    from .config import ConfigReader

__all__ = [
    "ConfigReader",
    "create_logger",
    "LabeledFrame",
    "import_object",
    "read_results",
    "restructure_results",
]


def __getattr__(name):
    # ConfigReader needs hydra, an optional "flow" dependency, so it is imported on
    # first use. The datasets only need the other helpers.
    if name == "ConfigReader":
        from .config import ConfigReader

        return ConfigReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
