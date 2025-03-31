from typing import Any, TypedDict

Options = TypedDict("Options", {"warnings": bool})

__options = Options(warnings=True)


def set_option(key: str, value: Any):
    """
    Sets the value of the specified option.

    Available options:
        - 'warnings': bool (default: True)
            If True, warnings will be shown.

    Args:
        key:   The option to set. Available options are 'warnings'.
        value: The value to set the option to.

    Raises:
        KeyError: If the key is not in the available options.

    Examples:
        >>> from seismostats.utils._config import set_option

        >>> # Disable warnings
        >>> set_option('warnings', False)

    See also:
        :func:`seismostats.utils._config.get_option`

    """
    global __options  # noqa
    if key in __options:
        __options[key] = value
    else:
        raise KeyError(f'Key "{key}" not in config.')


def get_option(key: str) -> Any:
    """
    Gets the value of the specified option.

    Available options:
        - 'warnings': bool (default: True)
            If True, warnings will be shown.

    Args:
        key:   The option to get. Available options are 'warnings'.

    Returns:
        value:  The value of the option.

    Examples:
        >>> from seismostats.utils._config import get_option

        >>> # Get the value of the 'warnings' option
        >>> warnings = get_option('warnings')
    """

    return __options[key]
