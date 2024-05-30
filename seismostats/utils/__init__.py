import functools

import pandas as pd
from jinja2 import Template, select_autoescape
from seismostats.utils.binning import bin_to_precision, get_cum_fmd, get_fmd


def _check_required_cols(df: pd.DataFrame,
                         required_cols: list[str]):
    """
    Check if a DataFrame has the required columns.

    Args:
        df : pandas DataFrame
            DataFrame to check.
            required_cols : list of str
            List of required columns.

    Returns:
        bool
            True if the DataFrame has the required columns, False otherwise.
    """

    if not set(required_cols).issubset(set(df.columns)):
        return False
    return True


def require_cols(_func=None, *,
                 require: list[str],
                 exclude: list[str] = None):
    """
    Decorator to check if a Class has the required columns for a method.

    Args:
        _func : function, optional
            Function to decorate.
        require : list of str
            List of required columns.
        exclude : list of str, optional
            List of columns to exclude from the required columns.
    """
    def decorator_require(func):
        @functools.wraps(func)
        def wrapper_require(self, *args, **kwargs):
            nonlocal require
            if exclude:
                require = [col for col in require if col not in exclude]
            if not _check_required_cols(self, require):
                raise AttributeError(
                    'Catalog is missing the following columns '
                    f'for execution of the method "{func.__name__}": '
                    f'{set(require).difference(set(self.columns))}.')
            value = func(self, *args, **kwargs)
            return value
        return wrapper_require

    if _func is None:
        return decorator_require
    else:
        return decorator_require(_func)


def _render_template(data: dict, template_path: str) -> str:
    with open(template_path) as t:
        template = Template(t.read(), autoescape=select_autoescape())

    qml = template.render(**data)
    return qml
