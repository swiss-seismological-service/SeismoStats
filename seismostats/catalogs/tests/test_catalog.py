import os
import re
import uuid

import numpy as np
import pandas as pd
import pytest
import datetime as dt
import inspect
import matplotlib.pyplot as plt

from seismostats.analysis.bvalue import estimate_b
from seismostats.catalogs.catalog import (CATALOG_COLUMNS, Catalog,
                                          ForecastCatalog)
from seismostats.utils.binning import bin_to_precision
from seismostats.plots.seismicity import plot_in_space
from seismostats.plots.basics import (plot_cum_count, plot_mags_in_time,
                                      plot_cum_fmd, plot_fmd)
from seismostats.plots.statistical import plot_mc_vs_b

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
        CATALOG_COLUMNS.sort()

    # Test inplace stripping
    catalog.strip(inplace=True)
    assert catalog.columns.tolist().sort() == CATALOG_COLUMNS.sort()

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
        (np.array([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6]),
         0.2),
        ([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6],
         0.2)
    ]
)
def test_catalog_bin(mag_values: np.ndarray, delta_m: float):
    catalog = Catalog({'magnitude': mag_values})

    with pytest.raises(ValueError):
        catalog.bin_magnitudes(delta_m=None)
    with pytest.raises(ValueError):
        catalog.bin_magnitudes()

    assert (catalog.bin_magnitudes(
        delta_m)['magnitude'].tolist()
        == bin_to_precision(mag_values, delta_m)).all()
    catalog_copy = Catalog({'magnitude': mag_values})
    catalog_copy.delta_m = 0.1
    assert (catalog_copy.bin_magnitudes()['magnitude'].tolist()
            == bin_to_precision(mag_values, 0.1)).all()

    return_value = catalog.bin_magnitudes(delta_m, inplace=True)
    assert (catalog['magnitude'].tolist()
            == bin_to_precision(mag_values, delta_m)).all()
    assert return_value is None
    assert catalog.delta_m == delta_m


def test_catalog_estimate_mc():
    # TODO once global_mc method is implemented
    catalog = Catalog({'magnitude': [0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]})

    with pytest.raises(ValueError):
        catalog.estimate_mc()


@pytest.mark.parametrize(
    "mag_values, delta_m, mc",
    [
        (np.array([0.0, 0.235, 0.238, 4.499, 4.5, 6, 0.1, 1.6]),
         0.001, 0.0)
    ]
)
def test_catalog_estimate_b(mag_values, delta_m, mc):
    catalog = Catalog({'magnitude': mag_values})

    b_value = estimate_b(catalog['magnitude'],
                         mc=mc,
                         delta_m=delta_m)
    return_value = catalog.estimate_b(mc=mc, delta_m=delta_m)
    assert catalog.b_value == b_value
    assert return_value == b_value

    catalog.mc = mc
    catalog.delta_m = delta_m
    return_value = catalog.estimate_b()
    assert catalog.b_value == b_value
    assert return_value == b_value


@pytest.fixture
def catalog_example():
    times = [dt.datetime(2020, 1, i) for i in range(1, 10)]
    magnitudes = [3.5, 3.6, 4.0, 3.8, 3.9, 6.0, 5.9, 4.2, 4.1]
    latitudes = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]
    longitudes = [120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0]
    cat = Catalog({"time": times, 
                   "magnitude": magnitudes,
                   "latitude": latitudes,
                   "longitude": longitudes})
    cat.delta_m = 0.1
    cat.mc = 3.5
    cat.b_value = 1.0

    return cat


@pytest.fixture
def mcs_for_plot_mc_vs_b():
    return [3.0, 3.5, 4.0]


def extract_names_and_default_values(parameters, exclude_args):
    params = {}
    for param, details in parameters.items():
        if param in exclude_args:
            continue
        if details.default == inspect.Parameter.empty:
            default_value = None
        else:
            default_value = details.default
        params[param] = default_value
    
    return params


def compare_method_and_function(method,
                                function,
                                exclude_args,
                                method_kwargs,
                                **kwargs):
    method_args = inspect.signature(method).parameters
    function_args = inspect.signature(function).parameters
    method_params = extract_names_and_default_values(method_args,
                                                     exclude_args)
    function_params = extract_names_and_default_values(function_args,
                                                       exclude_args)
    assert method_params == function_params

    method_output = method(**method_kwargs)
    assert isinstance(method_output, plt.Axes)

    function_output = function(**kwargs)
    assert isinstance(function_output, plt.Axes)

    assert type(method_output) == type(function_output)


@pytest.mark.parametrize("method, function, exclude_args, other_args", [
    ("plot_in_space", plot_in_space,
     ["magnitudes", "latitudes", "longitudes"], []),
    ("plot_cum_count", plot_cum_count,
     ["magnitudes", "times"], ["delta_m"]),
    ("plot_mags_in_time", plot_mags_in_time,
     ["magnitudes", "times"], []),
    ("plot_cum_fmd", plot_cum_fmd,
     ["magnitudes"], ["delta_m", "mc", "b_value"]),
    ("plot_fmd", plot_fmd,
     ["magnitudes"], ["delta_m"]),
    ("plot_mc_vs_b", plot_mc_vs_b,
     ["magnitudes"], ["delta_m", "mcs"])
])
def test_catalog_methods(catalog_example,
                         mcs_for_plot_mc_vs_b,
                         method,
                         function,
                         exclude_args,
                         other_args): 
    method_ref = getattr(catalog_example, method)
    kwargs_dict = {}
    method_kwargs = {}
    arg_to_value_map = {
        "magnitudes": catalog_example.magnitude,
        "latitudes": catalog_example.latitude,
        "longitudes": catalog_example.longitude,
        "times": catalog_example.time,
        "delta_m": catalog_example.delta_m,
        "mc": catalog_example.mc,
        "b_value": catalog_example.b_value
    }
    for arg in exclude_args + other_args:
        if arg in arg_to_value_map:
            kwargs_dict[arg] = arg_to_value_map[arg]
        elif arg == "mcs":
            kwargs_dict[arg] = mcs_for_plot_mc_vs_b
            other_args.remove("mcs")
            method_kwargs[arg] = mcs_for_plot_mc_vs_b
    exclude_args = ["self", *exclude_args, *other_args] 
    compare_method_and_function(method_ref,
                                function,
                                exclude_args,
                                method_kwargs,
                                **kwargs_dict)


def test_to_quakeml():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')
    with open(xml_file, 'r') as file:
        xml_content = file.read()

    catalog = Catalog.from_quakeml(
        xml_file,
        include_uncertainties=True,
        include_ids=True,
        include_quality=True)

    catalog_xml = catalog.to_quakeml(agencyID='SED', author='catalog-tools')
    catalog_xml = re.sub(r"[\n\t\s]*", "", catalog_xml)

    with open(xml_file, 'r') as file:
        xml = file.read()
    xml = re.sub(r"[\n\t\s]*", "", xml)

    assert catalog_xml == xml

    catalog2 = catalog.from_quakeml(
        xml_content,
        include_uncertainties=True,
        include_ids=True,
        include_quality=True)

    assert catalog.equals(catalog2)


def test_to_quakeml_without():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')

    catalog = Catalog.from_quakeml(xml_file)

    rgx = "(eventID|originID|magnitudeID)$"
    cols = catalog.filter(regex=rgx).columns
    assert len(cols) == 0

    rgx = "(_uncertainty|_lowerUncertainty|" \
        "_upperUncertainty|_confidenceLevel)$"
    cols = catalog.filter(regex=rgx).columns
    assert len(cols) == 0

    catalog = catalog._create_ids()
    event = catalog.iloc[0]
    event2 = catalog.iloc[1]
    assert event['eventID'] != event2['eventID'], "eventID should be unique"
    assert uuid.UUID(str(event['magnitudeID']))
    assert uuid.UUID(str(event['originID']))
    assert uuid.UUID(str(event['eventID']))
    assert event['magnitudeID'] == event['magnitude_MLhc_magnitudeID']
    assert event['magnitudeID'] != event['magnitude_MLv_magnitudeID']
    assert uuid.UUID(str(event['magnitude_MLv_magnitudeID']))


def test_to_quakeml_forecast():
    xml_file = os.path.join(PATH_RESOURCES, 'quakeml_data.xml')

    catalog1 = Catalog.from_quakeml(
        xml_file,
        include_uncertainties=True,
        include_ids=True,
        include_quality=True)
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
    assert catalog.columns.tolist() == CATALOG_COLUMNS

    catalog = Catalog.from_dict({})
    assert catalog.empty
    assert catalog.columns.tolist() == CATALOG_COLUMNS

    catalog = Catalog.from_dict({'magnitude': []}, include_ids=False)
    assert isinstance(catalog, Catalog)
