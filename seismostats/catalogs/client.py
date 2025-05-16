from datetime import datetime

import pandas as pd
import requests

from seismostats import Catalog
from seismostats.io.parser import parse_quakeml_response


class FDSNWSEventClient():
    """
    Client for downloading earthquake catalogs from the
    FDSNWS event service.

    Args:
        url:    base url of the FDSNWS event service
                (eg. https://earthquake.usgs.gov/fdsnws/event/1/query)
    """

    def __init__(self, url: str):
        self.url = url
        self.params = {}

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
                   include_quality: bool = False,
                   batch_size: int | None = None,
                   **kwargs) -> pd.DataFrame:
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
            batch_size:         If set to None, will download all events
                            in one request. If set, will download events
                            in batches of `batch_size`, but return the entire
                            catalog.
            kwargs:             Additional parameters to be passed to the
                            FDSNWS event service.

        Returns:
            catalog: The catalog as a Catalog Object.

        Examples:
            .. code-block:: python

                >>> from seismostats import FDSNWSEventClient
                >>> from datetime import datetime
                >>> url = 'http://eida.ethz.ch/fdsnws/event/1/query'
                >>> client = FDSNWSEventClient(url)
                >>> df = client.get_events(
                ...     start_time=datetime(2020, 1, 1),
                ...     end_time=datetime(2022, 1, 1),
                ...     min_magnitude=0.5,
                ...     min_longitude=5,
                ...     max_longitude=11,
                ...     min_latitude=45,
                ...     max_latitude=48)
                >>> df

                event_type time                latitude  longitude magnitude
                0  earthquake 2021-12-30 07:43:14 46.051 7.388  2.510  ...
                1  earthquake 2021-12-30 01:35:37 46.778 9.476  1.352  ...
                2  earthquake 2021-12-29 08:48:59 47.779 7.722  0.817  ...
                3  earthquake 2021-12-29 00:14:32 47.715 7.634  1.252  ...
                4  earthquake 2021-12-28 11:51:38 45.752 7.080  0.897  ...
                ...       ...                 ...    ...   ...    ...  ...

        """

        date_format = "%Y-%m-%dT%H:%M:%S"
        self.params = {}

        if start_time is not None:
            self.params['starttime'] = start_time.strftime(date_format)
        if end_time is not None:
            self.params['endtime'] = end_time.strftime(date_format)
        if min_latitude is not None:
            self.params['minlatitude'] = min_latitude
        if max_latitude is not None:
            self.params['maxlatitude'] = max_latitude
        if min_longitude is not None:
            self.params['minlongitude'] = min_longitude
        if max_longitude is not None:
            self.params['maxlongitude'] = max_longitude
        if min_magnitude is not None and delta_m is not None:
            self.params['minmagnitude'] = min_magnitude - (delta_m / 2)
        elif min_magnitude is not None:
            self.params['minmagnitude'] = min_magnitude
        if max_magnitude is not None:
            self.params['maxmagnitude'] = max_magnitude
        if include_all_magnitudes is not None:
            self.params['includeallmagnitudes'] = include_all_magnitudes
        if event_type is not None:
            self.params['eventtype'] = event_type
        self.params.update(kwargs)

        catalog = []

        if batch_size is not None:
            catalog = self._get_events_batched(batch_size, include_quality)
        else:
            r = requests.get(self.url, stream=True, params=self.params)
            catalog = parse_quakeml_response(r, include_quality=include_quality)

        catalog = Catalog.from_dict(
            catalog, include_uncertainty, include_ids)

        if start_time:
            catalog.starttime = start_time
        if end_time:
            catalog.endtime = end_time

        return catalog

    def _get_batch_params(self, batch_size: int) -> dict:
        batch_params = []
        for i in range(0, self._get_batch_count(batch_size)):
            offset = 1 + (i * batch_size)
            params = self.params | {'limit': batch_size, 'offset': offset}
            batch_params.append(params)
        return batch_params

    def _get_events_batched(self, batch_size: int, include_quality: bool
                            ) -> list[dict]:
        """
        Download the events in batches of `batch_size` in order to avoid
        timeouts.
        """
        catalog = []
        batch_params = self._get_batch_params(batch_size)
        for params in batch_params:
            r = requests.get(self.url, stream=True, params=params)
            batch = parse_quakeml_response(r, include_quality=include_quality)
            catalog.extend(batch)

        return catalog

    def _exists_with_offset(self, limit: int, offset: int) -> bool:
        """
        Check if there are events with the given offset. Can be used
        to find the number of events available.
        """
        params = self.params | {'limit': limit, 'offset': offset}
        r = requests.get(self.url, params=params)
        return (r.status_code == 200 and r.content)

    def _get_batch_count(self, batch_size: int) -> int:
        """
        Get the number of batches available, given a certain batch size.
        """
        chunk = 1
        while self._exists_with_offset(1, (chunk * batch_size) + 1):
            chunk *= 2

        low = chunk // 2
        high = chunk
        last_valid_chunk = 0

        while low <= high:
            mid = (low + high) // 2
            offset = (mid * batch_size) + 1
            if self._exists_with_offset(1, offset):
                last_valid_chunk = mid
                low = mid + 1
            else:
                high = mid - 1

        return last_valid_chunk + 1
