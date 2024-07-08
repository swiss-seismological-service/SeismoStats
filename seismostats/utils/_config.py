from typing import Any, TypedDict

Options = TypedDict('Options', {'warnings': bool})

__options = Options(warnings=True)


def set_option(key: str, value: Any):
    """
    Sets the value of the specified option.

        Available options:
            - 'warnings': bool (default: True)
                If True, warnings will be shown.

    Args:
        key :   The option to set.
        value : The value to set the option to.

    Raises:
        KeyError: If the key is not in the available options.

    """
    global __options
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
        key :   The option to get.

    Returns:
        value:  The value of the option.
    """

    return __options[key]
