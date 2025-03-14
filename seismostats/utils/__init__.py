# flake8: noqa
import functools
import math

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from seismostats.utils.binning import (
    bin_to_precision,
    get_cum_fmd,
    get_fmd,
    normal_round,
    normal_round_to_int,
    binning_test
)
from seismostats.utils.coordinates import CoordinateTransformer
from seismostats.utils.coordinates import (
    bounding_box_to_polygon,
    polygon_to_bounding_box,
)
from seismostats.utils.filtering import cat_intersect_polygon

from seismostats.utils.simulate_distributions import (
    simulate_magnitudes,
    simulate_magnitudes_binned,
)

from seismostats.utils._config import (
    get_option,
    set_option
)


def _check_required_cols(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """
    Check if a DataFrame has the required columns.

    Args:
        df:             DataFrame to check.
        required_cols:  List of required columns.

    Returns:
        check:          True if the DataFrame has the required columns,
                    False otherwise.
    """

    if not set(required_cols).issubset(set(df.columns)):
        return False
    return True


def require_cols(_func=None, *, require: list[str], exclude: list[str] = None):
    """
    Decorator to check if a Class has the required columns for a method.

    Args:
        _func:          Function to decorate.
        require:        List of required columns.
        exclude:        List of columns to exclude from the required columns.

    Returns:
        requirement:    Decorator function.
    """

    def decorator_require(func):
        @functools.wraps(func)
        def wrapper_require(self, *args, **kwargs):
            nonlocal require
            if exclude:
                require = [col for col in require if col not in exclude]
            if not _check_required_cols(self, require):
                raise AttributeError(
                    "Catalog is missing the following columns "
                    f'for execution of the method "{func.__name__}": '
                    f"{set(require).difference(set(self.columns))}."
                )
            value = func(self, *args, **kwargs)
            return value

        return wrapper_require

    if _func is None:
        return decorator_require
    else:
        return decorator_require(_func)


def is_nan(value: float) -> bool:
    """
    Check if a value is NaN.

    Args:
        value:  Value to check.

    Returns:
        check:  True if the value is NaN, False otherwise.

    Examples:
        >>> from seismostats.utils import is_nan

        >>> is_nan(float('nan'))
        True
        >>> is_nan(1.0)
        False
    """
    return isinstance(value, float) and math.isnan(value)


def _render_template(data: dict, template_path: str) -> str:
    """
    Render a Jinja template with the provided data.
    """
    env = Environment(
        loader=FileSystemLoader("/"),  # Base directory for templates
        autoescape=select_autoescape(),
    )
    env.tests["nan"] = is_nan

    template = env.get_template(template_path)

    qml = template.render(**data)
    return qml
