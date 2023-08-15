import functools

import pandas as pd
from geopandas import GeoDataFrame

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
    def get_magnitude(self):
        return self['magnitude'].values


def main():
    df_no = Catalog([{'a': 3, 'b': 2, 'c': 3}], name='cat')

    df_yes = Catalog([{'longitude': 1, 'latitude': 2, 'depth': 3,
                       'time': 4, 'magnitude': 5, 'magnitude_type': 6,
                       'magnitude_Mw': 7}], name='cat')

    print(type(df_yes))

    # df['a'] = [1, 2, 3]
    # df['b'] = [4, 5, 6]
    df_no['d'] = [7]

    print(df_no.get_magnitude())
    # print(df_yes)
    # df2 = df[['a', 'b']]
    # print(type(df))
    # print(df)
    # print(type(df2))
    # print(df2)

    # gdf = GeoDataFrame()
    # print(type(gdf))
    # gdf = gdf.from_dict([{'a': 3, 'b': 2, 'c': 3}])
    # print(gdf.area)
    # print(type(gdf))
    # print(type(gdf[['a', 'b']]))

    # url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
    # url = 'http://arclink.ethz.ch/fdsnws/event/1/query'
    # client = FDSNWSEventClient(url)
    # cat = client.get_events(
    #     start_time=datetime(2010, 7, 1),
    #     end_time=datetime(2020, 7, 1),
    #     include_all_magnitudes=False)


if __name__ == '__main__':
    main()
