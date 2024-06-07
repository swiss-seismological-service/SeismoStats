import os
import re
import uuid
import pytest

import numpy as np
import pandas as pd

from seismostats.seismicity.catalog import (REQUIRED_COLS_CATALOG, Catalog,
                                            ForecastCatalog)
from seismostats.utils.binning import bin_to_precision

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


@pytest.mark.parametrize(
    "mag_values, delta_m",
    [
        (np.array([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]),
         0.1),
        (np.array([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]),
         None),
        (np.array([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6]),
         0.2),
        ([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6],
         0.2)
    ]
)
def test_catalog_bin(mag_values: np.ndarray, delta_m: float):
    catalog = Catalog({'magnitude': mag_values})

    assert (catalog.bin_magnitudes(
        delta_m)['magnitude'].tolist()
        == bin_to_precision(mag_values, delta_m)).all()

    return_value = catalog.bin_magnitudes(delta_m, inplace=True)
    assert (catalog['magnitude'].tolist()
            == bin_to_precision(mag_values, delta_m)).all()
    assert return_value is None

    assert catalog.delta_m == delta_m


@pytest.mark.parametrize(
    "mag_values, delta_m",
    [
        (np.array([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]),
         0),
    ]
)
def test_catalog_bin_none(mag_values: np.ndarray, delta_m: float):
    catalog = Catalog({'magnitude': mag_values})

    with pytest.raises(ValueError):
        catalog.bin_magnitudes(delta_m=delta_m)


def test_catalog_estimate_mc():
    catalog = Catalog({'magnitude': [0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]})

    with pytest.raises(ValueError):
        catalog.estimate_mc()


def test_catalog_estimate_b():
    catalog = Catalog({'magnitude': [0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]})

    with pytest.raises(ValueError):
        catalog.estimate_b(mc=None, delta_m=None)
        catalog.estimate_b(mc=1.0, delta_m=None)
        catalog.estimate_b(mc=None, delta_m=0.1)


def test_to_quakeml():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')
    with open(xml_file, 'r') as file:
        xml_content = file.read()

    catalog = Catalog.from_quakeml(
        xml_file, includeuncertainties=True, includeids=True)
    catalog_xml = catalog.to_quakeml(agencyID='SED', author='catalog-tools')
    catalog_xml = re.sub(r"[\n\t\s]*", "", catalog_xml)

    with open(xml_file, 'r') as file:
        xml = file.read()
    xml = re.sub(r"[\n\t\s]*", "", xml)

    assert catalog_xml == xml

    catalog2 = catalog.from_quakeml(
        xml_content, includeuncertainties=True, includeids=True)
    assert catalog.equals(catalog2)


def test_to_quakeml_without():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')

    catalog = Catalog.from_quakeml(xml_file)

    rgx = "(eventid|originid|magnitudeid)$"
    cols = catalog.filter(regex=rgx).columns
    assert len(cols) == 0

    rgx = "(_uncertainty|_lowerUncertainty|" \
        "_upperUncertainty|_confidenceLevel)$"
    cols = catalog.filter(regex=rgx).columns
    assert len(cols) == 0

    catalog = catalog._create_ids()
    event = catalog.iloc[0]
    assert uuid.UUID(str(event['magnitudeid']))
    assert uuid.UUID(str(event['originid']))
    assert uuid.UUID(str(event['eventid']))
    assert event['magnitudeid'] == event['magnitude_MLhc_magnitudeid']
    assert event['magnitudeid'] != event['magnitude_MLv_magnitudeid']
    assert uuid.UUID(str(event['magnitude_MLv_magnitudeid']))


def test_to_quakeml_forecast():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')

    catalog1 = Catalog.from_quakeml(
        xml_file, includeuncertainties=True, includeids=True)
    catalog1.name = 'Catalog 1'
    catalog2 = catalog1.copy()
    catalog2.name = 'Catalog 2'

    catalog1['catalog_id'] = 1
    catalog2['catalog_id'] = 2

    catalog = pd.concat([catalog1, catalog2]).reset_index(drop=True)
    assert catalog.name == 'Catalog 1'

    catalog_xml = catalog.to_quakeml(agencyID='SED', author='catalog-tools')

    assert len(catalog_xml) == 2

    catalog_xml = catalog_xml[0]
    catalog_xml = re.sub(r"[\n\t\s]*", "", catalog_xml)

    with open(xml_file, 'r') as file:
        xml = file.read()
    xml = re.sub(r"[\n\t\s]*", "", xml)

    assert catalog_xml == xml

    catalog = pd.merge(catalog1, catalog2, how='outer')
    assert catalog.name == 'Catalog 1'


def test_empty_catalog():
    catalog = Catalog()
    assert catalog.empty
    assert catalog.columns.tolist() == REQUIRED_COLS_CATALOG

    catalog = Catalog.from_dict({})
    assert catalog.empty
    assert catalog.columns.tolist() == REQUIRED_COLS_CATALOG

    catalog = Catalog.from_dict({'magnitude': []}, includeids=False)
    assert isinstance(catalog, Catalog)
