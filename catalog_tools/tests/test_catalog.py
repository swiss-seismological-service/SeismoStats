import pandas as pd
import pytest

from catalog_tools.catalog import (REQUIRED_COLS, Catalog, ForecastCatalog,
                                   require_cols)
from catalog_tools.utils.binning import bin_to_precision

RAW_DATA = {'name': ['Object 1', 'Object 2', 'Object 3'],
            'magnitude': [10.0, 12.5, 8.2],
            'longitude': [120.0, 121.0, 122.0],
            'latitude': [30.0, 31.0, 32.0],
            'depth': [10.0, 11.0, 12.0],
            'time': [pd.Timestamp('2020-01-01'),
                     pd.Timestamp('2020-01-02'),
                     pd.Timestamp('2020-01-03')],
            'magnitude_type': ['Mw', 'Mw', 'Mw'],
            'ra': [120.0, 121.0, 122.0],
            'dec': [30.0, 31.0, 32.0],
            'catalog_id': [1, 1, 2]}


def test_catalog_init():
    # Test initialization with data
    data = {'name': ['Object 1', 'Object 2', 'Object 3'],
            'magnitude': [10.0, 12.5, 8.2]}
    catalog = Catalog(data)
    assert isinstance(catalog, Catalog)
    assert catalog.name is None

    # Test initialization with name
    catalog = Catalog(data, name='My Catalog')
    assert catalog.name == 'My Catalog'

    # Test initialization with additional arguments
    catalog = Catalog(data, columns=['name', 'magnitude'])


def test_forecast_catalog_init():
    # Test initialization with data
    data = {'name': ['Object 1', 'Object 2', 'Object 3'],
            'magnitude': [10.0, 12.5, 8.2]}
    catalog = ForecastCatalog(data)
    assert isinstance(catalog, ForecastCatalog)


def test_catalog_strip():
    # Test stripping columns
    catalog = Catalog(RAW_DATA)
    stripped_catalog = catalog.strip()
    assert isinstance(stripped_catalog, Catalog)
    assert stripped_catalog.columns.tolist().sort() == \
        REQUIRED_COLS.sort()

    # Test inplace stripping
    catalog.strip(inplace=True)
    assert catalog.columns.tolist().sort() == REQUIRED_COLS.sort()

    # Test constructor fallback
    dropped = catalog.drop(columns=['magnitude'])
    assert not isinstance(dropped, Catalog)


def test_forecast_catalog_strip():
    required_cols = REQUIRED_COLS + ['catalog_id']
    # Test stripping columns
    catalog = ForecastCatalog(RAW_DATA)
    stripped_catalog = catalog.strip()
    assert isinstance(stripped_catalog, ForecastCatalog)
    assert stripped_catalog.columns.tolist().sort() == \
        required_cols.sort()

    # Test inplace stripping
    catalog.strip(inplace=True)
    assert catalog.columns.tolist().sort() == required_cols.sort()

    # Test constructor fallback
    dropped = catalog.drop(columns=['magnitude'])
    assert not isinstance(dropped, ForecastCatalog)

    # Test constructor fallback "downgrade"
    dropped = catalog.drop(columns=['catalog_id'])
    assert isinstance(dropped, Catalog)


def test_catalog_bin():
    mag_values = [0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]
    delta_m = 0.1

    catalog = Catalog({'magnitude': mag_values})

    assert (catalog.bin_magnitudes(
        delta_m)['magnitude'].tolist()
        == bin_to_precision(mag_values, delta_m)).all()

    catalog.bin_magnitudes(delta_m, inplace=True)
    assert (catalog['magnitude'].tolist()
            == bin_to_precision(mag_values, delta_m)).all()


class TestCatalog:
    columns = ['name', 'magnitude']

    @require_cols(require=REQUIRED_COLS)
    def require(self):
        pass

    @require_cols(require=['name'])
    def require_spec(self):
        pass

    @require_cols(require=REQUIRED_COLS, exclude=['magnitude'])
    def require_exclude(self):
        pass

    def test_require(self):
        pytest.raises(AttributeError, self.require)

    def test_require_succeed(self):
        self.columns = REQUIRED_COLS
        self.require()

    def test_require_exclude(self):
        self.columns = REQUIRED_COLS
        self.columns.remove('magnitude')
        self.require_exclude()

    def test_require_spec(self):
        self.columns = ['name']
        self.require_spec()
