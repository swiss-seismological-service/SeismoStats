import functools

import pandas as pd

REQUIRED_COLS = ['longitude', 'latitude', 'depth',
                 'time', 'magnitude', 'magnitude_type']


def _check_required_cols(df, required_cols=REQUIRED_COLS):
    if not set(REQUIRED_COLS).issubset(set(df.columns)):
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

    _metadata = ['name']

    def __init__(self, data=None, *args, name=None, **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

    @property
    def _constructor(self):
        return _catalog_constructor_with_fallback

    @require_cols
    def strip(self, inplace=False):
        self.drop(columns=set(self.columns).difference(set(REQUIRED_COLS)),
                  inplace=inplace)
