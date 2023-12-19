import importlib
from typing import Callable, Union


def import_object(import_path: str) -> Union[object, Callable]:
    """Imports an object by their classpath.

    Allows to instantiate objects by configs, such as models, HPO objects, datasets,
    etc.

    Parameters
    ----------
    import_path : str
        The import path for the object to import.

    Returns
    -------
    object
        The imported object (this can be a class, a callable, a variable).
    """
    separator_idx = import_path.rindex(".")
    module_path = import_path[:separator_idx]
    obj_name = import_path[separator_idx + 1 :]

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
