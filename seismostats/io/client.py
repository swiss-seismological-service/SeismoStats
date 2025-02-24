from datetime import datetime

import pandas as pd
import requests

from seismostats import Catalog
from seismostats.io.parser import parse_quakeml_response


class FDSNWSEventClient():
    def __init__(self, url: str):
        """
        Client for downloading earthquake catalogs from the
        FDSNWS event service.

        Args:
            url:    base url of the FDSNWS event service
                    (eg. 'https://earthquake.usgs.gov/fdsnws/event/1/query')
        """
        self.url = url

    def get_events(self,
                   start_time: datetime | None = None,
                   end_time: datetime | None = None,
                   min_latitude: float | None = None,
                   max_latitude: float | None = None,
                   min_longitude: float | None = None,
                   max_longitude: float | None = None,
                   min_magnitude: float | None = None,
                   max_magnitude: float | None = None,
                   include_all_magnitudes: bool = False,
                   event_type: str | None = None,
                   delta_m: float | None = 0.1,
                   include_uncertainty: bool = False,
                   include_ids: bool = False,
                   include_quality: bool = False) -> pd.DataFrame:
        """Downloads an earthquake catalog based on a URL.

        Args:
            start_time:         Start time of the catalog.
            end_time:           End time of the catalog.
            min_latitude:       Minimum latitude of the catalog.
            max_latitude:       Maximum latitude of the catalog.
            min_longitude:      Minimum longitude of the catalog.
            max_longitude:      Maximum longitude of the catalog.
            min_magnitude:      Minimum magnitude of the catalog.
            max_magnitude:      Maximum magnitude of the catalog.
            include_all_magnitudes: Whether to include all magnitudes.
            event_type:         Filter by the type of events.
            delta_m:            Magnitude bin size. If >0, then events of
                            `magnitude >= (min_magnitude - delta_m/2)`
                            will be downloaded.
            include_uncertainty:    Whether to include uncertainty columns.
            include_ids:        Whether to include event,
                            magnitude and origin IDs.
            include_quality:    Whether to include quality columns.

        Returns:
            catalog: The catalog as a Catalog Object.

        Examples:
            Create a Catalog from a dictionary.

            >>> from seismostats.io import FDSNWSClient
            >>> from datetime import datetime
            >>> url = 'http://eida.ethz.ch/fdsnws/event/1/query'
            >>> client = FDSNWSClient(url)
            >>> df = client.get_events(
            ...     start_time=datetime(2020, 1, 1),
            ...     end_time=datetime(2022, 1, 1),
            ...     min_magnitude=0.5,
            ...     min_longitude=5,
            ...     max_longitude=11,
            ...     min_latitude=45,
            ...     max_latitude=48)
            >>> print(df)

               event_type time                latitude  longitude magnitude
            0  earthquake 2021-12-30 07:43:14 46.051445 7.388025  2.510115  ...
            1  earthquake 2021-12-30 01:35:37 46.778985 9.476219  1.352086  ...
            2  earthquake 2021-12-29 08:48:59 47.779511 7.722354  0.817480  ...
            3  earthquake 2021-12-29 00:14:32 47.715341 7.634432  1.252432  ...
            4  earthquake 2021-12-28 11:51:38 45.752843 7.080092  0.897306  ...
               ...        ...                 ...       ...       ...       ...
        """

        request_url = self.url + '?'
        date_format = "%Y-%m-%dT%H:%M:%S"

        if start_time is not None:
            request_url += f'&starttime={start_time.strftime(date_format)}'
        if end_time is not None:
            request_url += f'&endtime={end_time.strftime(date_format)}'
        if min_latitude is not None:
            request_url += f'&minlatitude={min_latitude}'
        if max_latitude is not None:
            request_url += f'&maxlatitude={max_latitude}'
        if min_longitude is not None:
            request_url += f'&minlongitude={min_longitude}'
        if max_longitude is not None:
            request_url += f'&maxlongitude={max_longitude}'
        if min_magnitude is not None and delta_m is not None:
            request_url += f'&minmagnitude={min_magnitude - (delta_m / 2)}'
        elif min_magnitude is not None:
            request_url += f'&minmagnitude={min_magnitude}'
        if max_magnitude is not None:
            request_url += f'&maxmagnitude={max_magnitude}'
        if include_all_magnitudes is not None:
            request_url += f'&includeallmagnitudes={include_all_magnitudes}'
        if event_type is not None:
            request_url += f'&eventtype={event_type}'

        catalog = []

        r = requests.get(request_url, stream=True)

        catalog = parse_quakeml_response(r, include_quality=include_quality)

        catalog = Catalog.from_dict(
            catalog, include_uncertainty, include_ids)

        if start_time:
            catalog.starttime = start_time
        if end_time:
            catalog.endtime = end_time

        return catalog
