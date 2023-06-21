import datetime as dt
import os
from unittest import mock
import numpy as np

from numpy.testing import assert_equal, assert_allclose, assert_array_less

from catalog_tools.download.download_catalogs import apply_edwards, \
    download_catalog_sed, prepare_sed_catalog, download_catalog_1


def test_apply_edwards():
    mag_types = ['MLh', 'MLhc', 'Ml', 'Mw']
    mags = [2.5, 3.5, 4.5, 5.5]

    assert_equal(
        [list(apply_edwards(typ, mag)) for (typ, mag) in zip(mag_types, mags)],
        [['Mw_converted', 2.506875], ['Mw_converted', 3.2734749999999995],
         ['Mw_converted', 4.138274999999999], ['Mw', 5.5]])


PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data')


def mocked_requests_get(*args, **kwargs):
    response = mock.MagicMock()
    response.getcode.return_value = 200

    with open(f'{PATH_RESOURCES}/catalog.csv', 'rb') as f:
        response.read.return_value = f.read()

    return response


@mock.patch('urllib.request.urlopen', side_effect=mocked_requests_get)
def test_download_catalog_1(mock_get):
    # download the CH catalog (mocked)
    base_query = 'http://its_mocked'
    min_mag = 3.0
    start_time = dt.datetime(1900, 1, 1)
    end_time = dt.datetime(2022, 1, 1)
    ch_cat = download_catalog_1(
        base_query='http://its_mocked',
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_mag)

    # check that the catalog is a dataframe
    assert_equal(str(type(ch_cat)), "<class 'pandas.core.frame.DataFrame'>")


@mock.patch('urllib.request.urlopen', side_effect=mocked_requests_get)
def test_download_catalog_sed(mock_get):
    # download the CH catalog (mocked)
    min_mag = 3.0
    start_time = dt.datetime(1900, 1, 1)
    end_time = dt.datetime(2022, 1, 1)
    ch_cat = download_catalog_sed(
        start_time=start_time, end_time=end_time, min_magnitude=min_mag)

    # check that the catalog was processed correctly
    assert_equal(
        [len(ch_cat), len(ch_cat.query("event_type != 'earthquake'"))],
        [1256, 0])


@mock.patch('urllib.request.urlopen', side_effect=mocked_requests_get)
def test_prepare_sed_catalog(mock_get):
    min_mag = 3.0
    start_time = dt.datetime(1900, 1, 1)
    end_time = dt.datetime(2022, 1, 1)
    delta_m = 0.1

    base_query = 'http://its_mocked'
    ch_cat = download_catalog_1(
        base_query=base_query,
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_mag)

    df = prepare_sed_catalog(
        ch_cat,
        delta_m=delta_m,
        only_earthquakes=True,
        convert_to_mw=True
    )

    unique_mags = np.unique(df["magnitude"])
    min_mag_diff = np.min(np.abs(np.diff(unique_mags)))

    converted_mags = df.query("mag_type == 'Mw_converted'")['magnitude']
    conv_diffs = ch_cat.loc[converted_mags.index, 'Magnitude'] - converted_mags
    min_conv_diff = np.min(np.abs(conv_diffs))

    assert_equal(
        [len(df), len(converted_mags)],
        [1256, 139])

    assert_allclose([delta_m], [min_mag_diff], rtol=1e-07)
    assert_array_less([0], [min_conv_diff])
