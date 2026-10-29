import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fairgbm import FairGBM
    from .fairlearn_classifier import FairlearnClassifier

__all__ = ["FairGBM", "FairlearnClassifier"]

# Each method is imported on first use, together with the library it wraps. A library
# that is missing or fails to load (e.g. fairgbm outside Linux x86-64) then only
# affects the method that uses it, not the rest of aequitas.flow.
_MODULES = {"FairGBM": ".fairgbm", "FairlearnClassifier": ".fairlearn_classifier"}


def __getattr__(name):
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
