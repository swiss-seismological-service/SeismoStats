import os
from datetime import datetime

import responses
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from responses import matchers

from seismostats import Catalog
from seismostats.catalogs.client import FDSNWSEventClient

PATH_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'query.xml')

date_format = "%Y-%m-%dT%H:%M:%S"


@responses.activate
def test_empty_catalog():
    """
    Tests the case where the Webservice returns an empty (204) response
    """

    url = 'http://mocked_url'
    responses.add(responses.GET, url, body='', status=204)

    client = FDSNWSEventClient(url)

    cat = client.get_events()

    assert cat.empty
    assert isinstance(cat, Catalog)


@responses.activate
def test_download_catalog():
    url = 'http://mocked_url'
    start_time = datetime(1900, 1, 1)
    end_time = datetime(2022, 1, 1)
    delta_m = 0.1
    min_mag = 3.0
    max_mag = 5.0
    min_lat = 45.0
    max_lat = 50.0
    min_lon = 0
    max_lon = 10.0
    event_type = 'earthquake'

    responses.add(responses.GET, url,
                  body=open(PATH_RESOURCES, 'rb'), status=200,
                  match=[
                      matchers.query_param_matcher(
                          {'starttime': start_time.strftime(date_format),
                           'endtime': end_time.strftime(date_format),
                           'minmagnitude': min_mag - (delta_m / 2),
                           'maxmagnitude': max_mag,
                           'minlatitude': min_lat,
                           'maxlatitude': max_lat,
                           'minlongitude': min_lon,
                           'maxlongitude': max_lon,
                           'eventtype': event_type,
                           'includeallmagnitudes': False})],)

    responses.add(responses.GET, url,
                  body=open(PATH_RESOURCES, 'rb'), status=200,
                  match=[
                      matchers.query_param_matcher(
                          {'includeallmagnitudes': True,
                           'minmagnitude': min_mag})],)

    client = FDSNWSEventClient(url)

    cat = client.get_events(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
        min_latitude=min_lat,
        max_latitude=max_lat,
        min_longitude=min_lon,
        max_longitude=max_lon,
        event_type=event_type,
    )

    assert len(cat) == 4

    assert isinstance(cat, Catalog)

    assert is_numeric_dtype(cat.magnitude.dtype)
    assert is_numeric_dtype(cat.longitude.dtype)
    assert is_numeric_dtype(cat.latitude.dtype)
    assert is_numeric_dtype(cat.depth.dtype)
    assert is_datetime64_any_dtype(cat.time.dtype)

    assert cat.columns.tolist().sort() == \
        ['evaluationMode', 'eventID', 'event_type', 'time', 'latitude',
         'longitude', 'depth', 'magnitude', 'magnitude_type'].sort()

    cat = client.get_events(
        include_all_magnitudes=True,
        include_uncertainty=True,
        delta_m=None,
        min_magnitude=min_mag
    )

    assert len(cat.columns) == 21
