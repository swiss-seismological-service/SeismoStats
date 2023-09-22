import os
import re

import pandas as pd

from catalog_tools.seismicity.catalog import (REQUIRED_COLS_CATALOG, Catalog,
                                              ForecastCatalog)
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

CATALOG_TEST_DATA = [
    {'depth': '1181.640625',
     'depth_uncertainty': '274.9552879',
     'event_type': 'earthquake',
     'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zqxyri',
     'latitude': '46.05144527',
     'latitude_uncertainty': '0.1222628824',
     'longitude': '7.388024848',
     'longitude_uncertainty': '0.1007121534',
     'magnitude': '2.510115344',
     'magnitude_MLhc': '2.510115344',
     'magnitude_MLhc_uncertainty': '0.23854491',
     'magnitude_MLv': '2.301758471',
     'magnitude_MLv_uncertainty': '0.2729312832',
     'magnitude_type': 'MLhc',
     'magnitude_uncertainty': '0.23854491',
     'time': '2021-12-30T07:43:14.681975Z'},
    {'depth': '3364.257812',
     'depth_uncertainty': '1036.395075',
     'event_type': 'earthquake',
     'eventid': 'smi:ch.ethz.sed/sc20a/Event/2021zihlix',
     'latitude': '47.37175484',
     'latitude_uncertainty': '0.1363265577',
     'longitude': '6.917056725',
     'longitude_uncertainty': '0.1277685645',
     'magnitude': '3.539687307',
     'magnitude_MLhc': '3.539687307',
     'magnitude_MLhc_uncertainty': '0.272435385',
     'magnitude_type': 'MLhc',
     'magnitude_uncertainty': '0.272435385',
     'time': '2021-12-25T14:49:40.125942Z'}]

PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data')


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
        REQUIRED_COLS_CATALOG.sort()

    # Test inplace stripping
    catalog.strip(inplace=True)
    assert catalog.columns.tolist().sort() == REQUIRED_COLS_CATALOG.sort()

    # Test constructor fallback
    dropped = catalog.drop(columns=['magnitude'])
    assert not isinstance(dropped, Catalog)


def test_forecast_catalog_strip():
    # Test stripping columns
    catalog = ForecastCatalog(RAW_DATA)
    stripped_catalog = catalog.strip()
    assert isinstance(stripped_catalog, ForecastCatalog)

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


def test_to_quakeml():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')

    catalog = Catalog(CATALOG_TEST_DATA)
    catalog_xml = catalog.to_quakeml()
    catalog_xml = re.sub(r"[\n\t\s]*", "", catalog_xml)

    with open(xml_file, 'r') as file:
        xml = file.read()
    xml = re.sub(r"[\n\t\s]*", "", xml)

    assert catalog_xml == xml
