import pandas as pd
from geopandas import GeoDataFrame

REQUIRED_COLS = ['longitude', 'latitude', 'depth',
                 'time', 'magnitude', 'magnitude_type']


def _check_required_cols(df):
    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        return False
    return True


def _catalog_constructor_with_fallback(*args, **kwargs):
    df = Catalog(*args, **kwargs)
    if not _check_required_cols(df):
        return pd.DataFrame(*args, **kwargs)
    return df


class Catalog(pd.DataFrame):

    _metadata = ['name']

    def __init__(self, data=None, *args, name=None, **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

    @property
    def _constructor(self):
        return _catalog_constructor_with_fallback


def main():
    # df = Catalog([{'a': 3, 'b': 2, 'c': 3}], name='cat')

    # # df['a'] = [1, 2, 3]
    # # df['b'] = [4, 5, 6]
    # # df['c'] = [7, 8, 9]

    # df2 = df[['a', 'b']]
    # print(type(df))
    # print(df)
    # print(type(df2))
    # print(df2)

    gdf = GeoDataFrame()
    gdf = gdf.from_dict([{'a': 3, 'b': 2, 'c': 3}])
    print(gdf.area)
    print(type(gdf[['a', 'b']]))


if __name__ == '__main__':
    main()
