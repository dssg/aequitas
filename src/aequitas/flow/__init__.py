from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment import DefaultExperiment, Experiment

__all__ = ["DefaultExperiment", "Experiment"]


def __getattr__(name):
    # The experiments need the optional "flow" dependencies (hpt, optuna, hydra...),
    # so they are imported on first use. This keeps submodules such as
    # aequitas.flow.datasets importable with the base installation.
    if name in __all__:
        from . import experiment

        return getattr(experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
