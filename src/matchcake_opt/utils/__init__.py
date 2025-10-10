import argparse


def get_all_subclasses(__class, include_base_cls: bool = False) -> set:  # pragma: no cover
    r"""
    Get all the subclasses of a class.

    :param __class: Class
    :type __class: Any
    :param include_base_cls: Include the base class in the set of subclasses
    :type include_base_cls: bool
    :return: Subclasses
    :rtype: set
    """
    subclasses: set = set({})
    for subclass in __class.__subclasses__():
        subclasses.add(subclass)
        subclasses |= get_all_subclasses(subclass)
    if include_base_cls:
        subclasses.add(__class)
    return subclasses


def set_args_with_kwargs(args: argparse.Namespace, *, ignore_errors=False, **kwargs):  # pragma: no cover
    """
    Set the attributes of the args object with the provided keyword arguments.
    This allows for dynamic setting of arguments based on the provided kwargs.
    """
    for key, value in kwargs.items():
        if not hasattr(args, key) and not ignore_errors:
            raise ValueError(f"Argument {key} is not recognized in the args object.")
        setattr(args, key, value)
    return args
