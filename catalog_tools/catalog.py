from __future__ import annotations

import functools

import pandas as pd

from catalog_tools.utils.binning import bin_to_precision

REQUIRED_COLS = ['longitude', 'latitude', 'depth',
                 'time', 'magnitude', 'magnitude_type']


def _check_required_cols(df: pd.DataFrame,
                         required_cols: list[str] = REQUIRED_COLS):
    if not set(required_cols).issubset(set(df.columns)):
        return False
    return True


def _catalog_constructor_with_fallback(*args, **kwargs):
    df = Catalog(*args, **kwargs)
    if not _check_required_cols(df):
        return pd.DataFrame(*args, **kwargs)
    return df


def require_cols(_func=None, *,
                 require: list[str] = REQUIRED_COLS,
                 exclude: list[str] = None):
    """
    Decorator to check if a Class has the required columns for a method.

    Args:
        _func : function, optional
            Function to decorate.
        require : list of str, optional
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


class Catalog(pd.DataFrame):
    """
    A subclass of pandas DataFrame that represents a catalog of items.

    Args:
        data : array-like, Iterable, dict, or DataFrame, optional
            Data to initialize the catalog with.
        name : str, optional
            Name of the catalog.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to pandas
            DataFrame constructor.

    Notes:
        The Catalog class is a subclass of pandas DataFrame, and inherits
        all of its methods and attributes.
    """
    _metadata = ['name']

    def __init__(self, data=None, *args, name=None, **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

    @property
    def _constructor(self):
        return _catalog_constructor_with_fallback

    @require_cols
    def strip(self, inplace: bool = False) -> Catalog | None:
        """
        Remove all columns except the required ones.

        Args:
            inplace : bool, optional
                If True, do operation inplace.

        Returns:
            Catalog or None
                If inplace is True, returns None. Otherwise, returns a new
                Catalog with the stripped columns.
        """
        df = self.drop(columns=set(self.columns).difference(set(REQUIRED_COLS)),
                       inplace=inplace)
        if not inplace:
            return df

    @require_cols(require=['magnitude'])
    def bin_magnitudes(
            self, delta_m: float, inplace: bool = False) -> Catalog | None:
        """
        Bin magnitudes to a given precision.

        Args:
            delta_m : float
                Magnitude bin size.
            inplace : bool, optional
                If True, do operation inplace.

        Returns:
            Catalog or None
        """
        if inplace:
            df = self
        else:
            df = self.copy()

        df['magnitude'] = bin_to_precision(df["magnitude"], delta_m)

        if not inplace:
            return df