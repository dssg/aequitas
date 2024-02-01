import importlib
from typing import Callable, Union
import inspect


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
    obj_name = import_path[separator_idx + 1:]

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def instantiate_object(class_object: Union[str, Callable], **kwargs) -> object:
    """Instantiates an object by their classpath.

    Parameters
    ----------
    class_object : Union[str, Callable]
        The classpath of the object to instantiate.
    **kwargs
        The keyword arguments to pass to the class constructor.

    Returns
    -------
    object
        The instantiated object.
    """

    class_object = import_object(class_object)
    signature = inspect.signature(class_object)
    if (
        signature.parameters[list(signature.parameters.keys())[-1]].kind
        == inspect.Parameter.VAR_KEYWORD
    ):
        args = kwargs
    else:
        args = {
            arg: value for arg, value in kwargs.items() if arg in signature.parameters
        }
    return class_object(**args)
