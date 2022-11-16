import datetime as dt
import os
from unittest import mock

from numpy.testing import assert_equal

from catalogue_tools.download.download_catalogues import (apply_edwards,
                                                          download_catalog_sed)


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
def test_download_catalogue_sed(mock_get):
    min_mag = 3.0
    start_time = dt.datetime(1900, 1, 1)
    end_time = dt.datetime(2022, 1, 1)

    # download the CH catalog
    ch_cat = download_catalog_sed(start_time=start_time, end_time=end_time,
                                  min_magnitude=min_mag,
                                  only_earthquakes=False)

    # check that the downloaded catalog is correct
    assert_equal(
        [len(ch_cat), len(ch_cat.query("event_type != 'earthquake'"))],
        [1274, 18])
