from io import BytesIO
import urllib.request
import datetime as dt
import pandas as pd
from typing import Optional
from obspy.clients.fdsn.client import Client
import numpy

# catalog tools
from catalog_tools.utils.binning import bin_to_precision


def apply_edwards(mag_type: str, mag: float) -> pd.Series:
    """
    Converts local magnitudes to moment magnitudes according to

    Edwards, B., Allmann, B., Fäh, D., & Clinton, J. (2010).
    Automatic computation of moment magnitudes for small earthquakes
    and the scaling of local to moment magnitude.
    Geophysical Journal International, 183(1), 407-420.
    """
    if "l" in mag_type.lower():
        return pd.Series(
            ['Mw_converted', 1.02 + 0.472 * mag + 0.0491 * mag ** 2])
    elif "w" in mag_type.lower():
        return pd.Series([mag_type, mag])


def download_catalog_sed(
        start_time: dt.datetime = dt.datetime(1970, 1, 1),
        end_time: dt.datetime = dt.datetime.now(),
        min_latitude: Optional[float] = None,
        max_latitude: Optional[float] = None,
        min_longitude: Optional[float] = None,
        max_longitude: Optional[float] = None,
        min_magnitude: float = 0.01,
        delta_m: float = 0.1
) -> pd.DataFrame:
    """Downloads the Swiss earthquake catalog.

    Args:
      start_time: start time of the catalog.
      end_time: end time of the catalog. defaults to current time.
      min_latitude: minimum latitude of catalog.
      max_latitude: maximum latitude of catalog.
      min_longitude: minimum longitude of catalog.
      max_longitude: maximum longitude of catalog.
      min_magnitude: minimum magnitude of catalog.
      delta_m: magnitude bin size. if >0, then
        events of magnitude >= (min_magnitude - delta_m/2) will be downloaded.

    Returns:
      The catalog as a pandas DataFrame.

    """
    base_query = 'http://arclink.ethz.ch/fdsnws/event/1/query?'
    st_tm = 'starttime=' + start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_tm = '&endtime=' + end_time.strftime("%Y-%m-%dT%H:%M:%S")
    min_mag = '&minmagnitude=' + str(min_magnitude - delta_m / 2)
    min_lat = '&minlatitude=' + str(
        min_latitude) if min_latitude is not None else None
    min_lon = '&minlongitude=' + str(
        min_longitude) if min_longitude is not None else None
    max_lat = '&maxlatitude=' + str(
        max_latitude) if max_latitude is not None else None
    max_lon = '&maxlongitude=' + str(
        max_longitude) if max_longitude is not None else None

    link = base_query + st_tm + end_tm + min_mag + ''.join(
        [part for part in [min_lat, min_lon, max_lat, max_lon] if
         part is not None]) + '&format=text'
    response = urllib.request.urlopen(link)
    data = response.read()

    df = pd.read_csv(BytesIO(data), delimiter="|")

    return df


def prepare_sed_catalog(
        df: pd.DataFrame,
        delta_m: float = 0.1,
        only_earthquakes: bool = True,
        convert_to_mw: bool = True
) -> pd.DataFrame:
    """Does standard treatment of the SED catalog after it has been downloaded.

    Args:
        df: downloaded catalog
        delta_m: magnitude bin size to be applied.
        only_earthquakes: if True, only
            events of event_type earthquake are kept.
        convert_to_mw: if True, local magnitudes are converted to Mw
            using Edwards et al.

    Returns:
        the catalog as a DataFrame
    """
    cat = df.copy()
    cat.rename({"Magnitude": "magnitude", "Latitude": "latitude",
                "Longitude": "longitude", "Time": "time", "Depth/km": "depth",
                "EventType": 'event_type', "MagType": 'mag_type'}, axis=1,
               inplace=True)

    if convert_to_mw:
        cat[['mag_type', 'magnitude']] = cat.apply(
            lambda x: apply_edwards(x['mag_type'], x['magnitude']), axis=1)

    cat["time"] = pd.to_datetime(cat["time"])
    cat.sort_values(by="time", inplace=True)

    if only_earthquakes:
        cat.query('event_type == "earthquake"', inplace=True)

    if delta_m > 0:
        cat["magnitude"] = bin_to_precision(cat["magnitude"], delta_m)

    return cat


def download_catalog(
        client_name='EMSC',
        starttime=dt.datetime(2023, 1, 1),
        endtime=dt.datetime.now(),
        minlatitude=None,
        maxlatitude=None,
        minlongitude=None,
        maxlongitude=None,
        minmagnitude=0,
) -> pd.DataFrame:

    client = Client(base_url=client_name)

    try:
        events = client.get_events(
            starttime=starttime,
            endtime=endtime,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            minmagnitude=minmagnitude,
        )
    except:

        start_1 = starttime
        mid_1 = starttime + (endtime - starttime)/2
        end_1 = endtime

        half_1 = client.get_events(
            starttime=start_1,
            endtime=mid_1,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            minmagnitude=minmagnitude,
        )
        half_2 = client.get_events(
            starttime=mid_1,
            endtime=end_1,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            minmagnitude=minmagnitude,
        )

        half_1.extend(half_2)

        events = half_1

    evs = []

    for event in events:
        lat = event.origins[0].latitude
        lon = event.origins[0].longitude
        depth = event.origins[0].depth / 1000
        time = event.origins[0].time
        time = dt.datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
        mag = event.magnitudes[0].mag
        mag_type = event.magnitudes[0].magnitude_type

        evs.append(pd.Series([time, lat, lon, depth, mag, mag_type]))

    cat = pd.DataFrame(evs)
    cat.columns = [
        'time', 'latitude', 'longitude', 'depth', 'magnitude', 'mag_type']
    cat.sort_values(by="time", inplace=True)
    cat.index = np.arange(len(cat))

    return cat
