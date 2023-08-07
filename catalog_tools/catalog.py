import pandas as pd

required_cols = ['longitude', 'latitude', 'depth',
                 'time', 'magnitude', 'magnitude_type']


def _catalog_constructor_with_fallback(*args, **kwargs):
    df = Catalog(*args, **kwargs)
    if not set(required_cols).issubset(set(df.columns)):
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
    df = Catalog(name='cat')

    df['a'] = [1, 2, 3]
    df['b'] = [4, 5, 6]
    df['c'] = [7, 8, 9]

    df2 = df[['a', 'b']]
    print(type(df))
    print(df)
    print(type(df2))
    print(df2)


if __name__ == '__main__':
    main()
