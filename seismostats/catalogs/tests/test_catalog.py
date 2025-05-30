import datetime as dt
import inspect
import os
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from typing_extensions import get_type_hints

from seismostats.analysis.avalue.positive import APositiveAValueEstimator
from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.bvalue.tests.test_bvalues import magnitudes
from seismostats.analysis.bvalue.utils import beta_to_b_value
from seismostats.analysis.estimate_mc import (estimate_mc_b_stability,
                                              estimate_mc_ks, estimate_mc_maxc)
from seismostats.analysis.tests.test_estimate_mc import KS_DISTS, MAGNITUDES
from seismostats.catalogs.catalog import (CATALOG_COLUMNS, Catalog,
                                          ForecastCatalog)
from seismostats.plots.basics import (plot_cum_count, plot_cum_fmd, plot_fmd,
                                      plot_mags_in_time)
from seismostats.plots.seismicity import plot_in_space
from seismostats.plots.statistical import plot_mc_vs_b
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


def test_catalog_init_datetime():
    mags = np.array([0, 0.9])
    times = np.array(['2000-01-01T00:00:00Z',
                      '2000-01-02T00:00:00.0000Z'])

    cat = Catalog({'magnitude': mags, 'time': times})
    assert isinstance(cat.time[0], pd.Timestamp)

    times = np.array(['250-01-01T00:00:00Z',
                      '1500-01-02T00:00:00.0000Z'])
    cat = Catalog({'magnitude': mags, 'time': times})
    assert isinstance(cat.time[0], datetime)


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


def extract_names_and_default_values(parameters, exclude_args):
    params = OrderedDict()
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

    assert type(method_output) is type(function_output)


@pytest.mark.filterwarnings("ignore::UserWarning")
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
     ["magnitudes"], ["fmd_bin"]),
    ("plot_mc_vs_b", plot_mc_vs_b,
     ["magnitudes"], ["delta_m", "mcs"])
])
def test_catalog_methods(catalog_example,
                         method,
                         function,
                         exclude_args,
                         other_args):
    mcs = [3.0, 3.5, 4.0]
    fmd_bin = 0.1
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
            kwargs_dict[arg] = mcs
            other_args.remove("mcs")
            method_kwargs[arg] = mcs
        elif arg == "fmd_bin":
            kwargs_dict[arg] = fmd_bin
            other_args.remove("fmd_bin")
            method_kwargs[arg] = fmd_bin
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


def test_catalog_estimate_mc():
    catalog = Catalog({'magnitude': [0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]})

    with pytest.raises(ValueError):
        catalog.estimate_mc_ks()


def function_signature_match(func1, func2):
    sig1 = inspect.signature(func1)
    sig2 = inspect.signature(func2)

    params1 = list(sig1.parameters.values())
    params2 = list(sig2.parameters.values())

    # Check if the number of parameters match
    assert len(params1) == len(params2), "Number of parameters do not match"

    # Check return type hints
    hints1 = get_type_hints(func1)
    hints2 = get_type_hints(func2)

    return1 = hints1.get('return', None)
    return2 = hints2.get('return', None)

    assert return1 == return2, \
        f"Return type hints differ:\n{return1}\n!=\n{return2}"

    # Second argument: name must match, default can differ
    assert params1[1].name == params2[1].name, \
        "Second argument names don't match"
    assert params1[1].kind == params2[1].kind, \
        "Second argument kinds don't match"

    # Remaining arguments: check name, kind, and default
    for i in range(2, len(params1)):
        p1 = params1[i]
        p2 = params2[i]

        assert p1.name == p2.name, \
            f"Param {i + 1} names don't match: {p1.name} != {p2.name}"
        assert p1.kind == p2.kind, \
            f"Param {p1.name} kind mismatch: {p1.kind} != {p2.kind}"
        assert p1.default == p2.default, \
            f"Param {p1.name} default mismatch: {p1.default} != {p2.default}"


def test_estimate_mc_ks_regression():
    func1 = Catalog.estimate_mc_ks
    func2 = estimate_mc_ks
    function_signature_match(func1, func2)


def test_estimate_mc_maxc_regression():
    func1 = Catalog.estimate_mc_maxc
    func2 = estimate_mc_maxc
    function_signature_match(func1, func2)


def test_estimate_mc_b_stability_regression():
    func1 = Catalog.estimate_mc_b_stability
    func2 = estimate_mc_b_stability
    function_signature_match(func1, func2)


def test_estimate_mc_functionality():
    cat = Catalog({'magnitude': MAGNITUDES})
    mcs = [0.8, 0.9, 1.0, 1.1]

    best_mc, mc_info = \
        cat.estimate_mc_ks(0.1,
                           mcs,
                           0.1,
                           b_value=beta_to_b_value(2.24),
                           ks_ds_list=KS_DISTS,
                           )

    assert_equal(1.1, best_mc)
    assert_equal(cat.mc, 1.1)

    assert_equal(beta_to_b_value(2.24), mc_info['best_b_value'])

    assert_allclose(
        [beta_to_b_value(2.24), beta_to_b_value(
            2.24), beta_to_b_value(2.24), beta_to_b_value(2.24)],
        mc_info['b_values_tested'],
        rtol=1e-7,
    )
    assert_allclose([0.0, 0.0, 0.0128, 0.4405], mc_info['p_values'], atol=0.03)
    assert_equal(mc_info['mcs_tested'], mcs)
    assert_allclose(mc_info['ks_ds'], [0.42931381663381224, 0.30109531596808387,
                    0.14068486563063504, 0.07052420897739642], rtol=1e-7)


def test_estimate_mc_maxc_functionality():
    cat = Catalog({'magnitude': MAGNITUDES})
    cat.estimate_mc_maxc(fmd_bin=0.1, correction_factor=0.2)
    assert_equal(1.3, cat.mc)


def test_estimate_mc_b_stability():
    cat = Catalog({'magnitude': MAGNITUDES})
    cat.estimate_mc_b_stability(delta_m=0.1, stability_range=0.5)
    assert_almost_equal(1.1, cat.mc)


@patch('seismostats.catalogs.catalog.estimate_mc_maxc',
       return_value=np.arange(0, 2))
def test_estimate_mc_maxc_catalog(mc_maxc_mock: MagicMock):
    cat = Catalog({'magnitude': MAGNITUDES})

    with pytest.raises(TypeError):
        cat.estimate_mc_maxc()

    cat.estimate_mc_maxc(fmd_bin=0.123)
    _, kwargs = mc_maxc_mock.call_args
    assert kwargs['fmd_bin'] == 0.123


@patch('seismostats.catalogs.catalog.estimate_mc_b_stability',
       return_value=np.arange(0, 2))
def test_estimate_mc_b_stability_catalog(mc_bvalue_mock: MagicMock):
    cat = Catalog({'magnitude': MAGNITUDES})

    with pytest.raises(ValueError):
        cat.estimate_mc_b_stability()

    cat.estimate_mc_b_stability(delta_m=0.123)
    _, kwargs = mc_bvalue_mock.call_args
    assert kwargs['delta_m'] == 0.123

    cat.delta_m = 0.321
    cat.estimate_mc_b_stability()
    _, kwargs = mc_bvalue_mock.call_args
    assert kwargs['delta_m'] == 0.321


@patch('seismostats.catalogs.catalog.estimate_mc_ks',
       return_value=np.arange(0, 2))
def test_estimate_mc_catalog(mc_ks_mock: MagicMock):
    cat = Catalog({'magnitude': MAGNITUDES})

    with pytest.raises(ValueError):
        cat.estimate_mc_ks()

    cat.estimate_mc_ks(delta_m=0.123)
    _, kwargs = mc_ks_mock.call_args
    assert kwargs['delta_m'] == 0.123

    cat.delta_m = 0.321
    cat.estimate_mc_ks()
    _, kwargs = mc_ks_mock.call_args
    assert kwargs['delta_m'] == 0.321

    cat.estimate_mc_ks(b_value=1.0)
    _, kwargs = mc_ks_mock.call_args
    assert kwargs['b_value'] == 1.0

    cat.b_value = 2.0
    cat.estimate_mc_ks()
    _, kwargs = mc_ks_mock.call_args
    assert kwargs['b_value'] == 2.0

    cat.estimate_mc_ks(b_value=1.0)
    _, kwargs = mc_ks_mock.call_args
    assert kwargs['b_value'] == 1.0


def test_estimate_b_functionality():
    delta_m = 0.1
    mags = bin_to_precision(magnitudes(1), delta_m)
    mc = 0
    b_est_correct = 0.9985052730956719
    weights = np.ones(len(mags))

    cat = Catalog({'magnitude': mags, 'weight': weights})
    cat.delta_m = delta_m
    cat.mc = mc

    b_estimate = cat.estimate_b()
    assert_almost_equal(b_est_correct, b_estimate.b_value)

    b_estimate_weighted = cat.estimate_b(weights=weights)
    assert_almost_equal(b_est_correct, b_estimate_weighted.b_value)


def test_estimate_b_method():
    delta_m = 0.1
    mc = 0
    mags = bin_to_precision(magnitudes(1), delta_m)
    mags = mags[mags >= mc - delta_m / 2]
    weights = np.ones(len(mags))

    cat = Catalog({'magnitude': mags, 'weight': weights})
    cat.delta_m = delta_m
    cat.mc = mc

    dmc = 0.3
    b_est_correct = 1.00768483769521
    b_est_correct_new = 1.009680716817806

    bmethod = BPositiveBValueEstimator

    b_estimate = cat.estimate_b(method=bmethod, dmc=dmc)
    assert_almost_equal(b_est_correct, b_estimate.b_value)

    new_weights = cat['weight'].copy()
    new_weights[0:100] = 0

    b_estimate = cat.estimate_b(method=bmethod, dmc=dmc, weights=new_weights)
    assert_almost_equal(b_est_correct_new, b_estimate.b_value)

    cat['weight'] = new_weights
    b_estimate = cat.estimate_b(method=bmethod, dmc=dmc)
    assert_almost_equal(b_est_correct_new, b_estimate.b_value)


def test_estimate_b_dynamic_col_arg():
    # test that time (or any other additional argument) is correctly used
    mags = np.array([0, 0.9, 0.5, 0.2, -1])
    times = np.array([datetime(2000, 1, 1),
                      datetime(2000, 1, 2),
                      datetime(2000, 1, 5),
                      datetime(2000, 1, 4),
                      datetime(2000, 1, 3)])

    cat = Catalog({'magnitude': mags, 'time': times})
    bmethod = BPositiveBValueEstimator
    estimator = cat.estimate_b(mc=-1, delta_m=0.1, method=bmethod, times=times)
    assert (mags[estimator.idx] == np.array([0.9, 0.2, 0.5])).all()

    estimator = cat.estimate_b(mc=-1, delta_m=0.1, method=bmethod)
    assert (mags[estimator.idx] == np.array([0.9, 0.2, 0.5])).all()


def test_estimate_b_catalog():
    mags = bin_to_precision(magnitudes(1), 0.1)

    estimator = MagicMock
    estimator._weights_supported = True
    estimator.calculate = MagicMock()

    cat = Catalog({'magnitude': mags})

    with pytest.raises(ValueError):
        cat.estimate_b()

    with pytest.raises(ValueError):
        cat.estimate_b(delta_m=0.1)

    with pytest.raises(ValueError):
        cat.estimate_b(mc=0)

    cat.estimate_b(mc=0, delta_m=0.123, method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (0, 0.123)

    cat.delta_m = 0.321
    cat.mc = 1
    cat.estimate_b(mc=0, delta_m=0.123, method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (0, 0.123)

    cat.estimate_b(method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (1, 0.321)

    estimator._weights_supported = False
    cat.estimate_b(method=estimator, weights=np.ones(len(mags)))
    _, kwargs = estimator.calculate.call_args
    assert kwargs['weights'] is None


def test_estimate_a_functionality():
    # should mainly test that the method is called correctly and
    # the two functions are consistent with each other.

    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    cat = Catalog({'magnitude': mags})

    estimator = cat.estimate_a(mc=1, delta_m=1)
    assert estimator.a_value == 1.0

    # reference magnitude is given and b-value given
    estimator = cat.estimate_a(mc=1, delta_m=0.0,
                               m_ref=0, b_value=1)
    assert estimator.a_value == 2.0

    # reference magnitude but no b-value
    with pytest.raises(ValueError):
        cat.estimate_a(mc=1, delta_m=0.0001, m_ref=0)

    # reference time is given
    estimator = cat.estimate_a(mc=1, delta_m=0.0001,
                               scaling_factor=10)
    assert estimator.a_value == 0.0


def test_estimate_a_method():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 12), timedelta(days=1)).astype(datetime)

    cat = Catalog({'magnitude': mags, 'time': times})

    amethod = APositiveAValueEstimator
    estimator = cat.estimate_a(mc=1, delta_m=1, method=amethod, times=times)
    assert_almost_equal(10**estimator.a_value, 10.0)

    # reference magnitude is given and b-value given
    estimator = cat.estimate_a(mc=1, delta_m=1,
                               m_ref=0, b_value=1, method=amethod)
    assert_almost_equal(10**estimator.a_value, 100.0)

    cat = Catalog({'magnitude': mags})
    estimator = cat.estimate_a(mc=1, delta_m=1, method=amethod, times=times)
    assert_almost_equal(10**estimator.a_value, 10.0)


def test_estimate_a_catalog():
    mags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10])
    times = np.arange(datetime(2000, 1, 1), datetime(
        2000, 1, 12), timedelta(days=1)).astype(datetime)

    estimator = MagicMock
    estimator.calculate = MagicMock()

    cat = Catalog({'magnitude': mags, 'time': times})

    with pytest.raises(ValueError):
        cat.estimate_a()

    with pytest.raises(ValueError):
        cat.estimate_a(mc=0)

    with pytest.raises(ValueError):
        cat.estimate_a(delta_m=0.1)

    cat.estimate_a(mc=0, delta_m=0.123, method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (0, 0.123)

    cat.delta_m = 0.321
    cat.mc = 1
    cat.estimate_a(mc=0, delta_m=0.123, method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (0, 0.123)

    cat.estimate_a(method=estimator)
    args, _ = estimator.calculate.call_args
    assert args[1:3] == (1, 0.321)
