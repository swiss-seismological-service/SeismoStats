from datetime import datetime
from xml.sax import handler, make_parser

import pandas as pd
import requests

from catalog_tools.catalog import Catalog
from catalog_tools.io.parser import QuakeMLHandler


class FDSNWSEventClient():
    """
    Args:
        url:    base url of the FDSNWS event service
                (eg. 'https://earthquake.usgs.gov/fdsnws/event/1/query')
    """

    def __init__(self, url: str):
        self.url = url

    def get_events(self, start_time: datetime | None = None,
                   end_time: datetime | None = None,
                   min_latitude: float | None = None,
                   max_latitude: float | None = None,
                   min_longitude: float | None = None,
                   max_longitude: float | None = None,
                   min_magnitude: float | None = None,
                   include_all_magnitudes: bool | None = None,
                   event_type: str | None = None,
                   delta_m: float = 0.1,
                   include_uncertainty: bool = False) -> pd.DataFrame:
        """Downloads an earthquake catalog based on a URL.

        Args:
            base_query:     base query url ()
            start_time:     start time of the catalog.
            end_time:       end time of the catalog. defaults to current time.
            min_latitude:   minimum latitude of catalog.
            max_latitude:   maximum latitude of catalog.
            min_longitude:  minimum longitude of catalog.
            max_longitude:  maximum longitude of catalog.
            min_magnitude:  minimum magnitude of catalog.
            include_all_magnitudes: whether to include all magnitudes.
            event_type:     type of event to download.
            delta_m:        magnitude bin size. if >0, then events of
                magnitude >= (min_magnitude - delta_m/2) will be downloaded.
            include_uncertainty: whether to include uncertainty columns.

        Returns:
            The catalog as a Catalog Object.

        """
        request_url = self.url + '?'
        date_format = "%Y-%m-%dT%H:%M:%S"

        if start_time:
            request_url += f'&starttime={start_time.strftime(date_format)}'
        if end_time:
            request_url += f'&endtime={end_time.strftime(date_format)}'
        if min_latitude:
            request_url += f'&minlatitude={min_latitude}'
        if max_latitude:
            request_url += f'&maxlatitude={max_latitude}'
        if min_longitude:
            request_url += f'&minlongitude={min_longitude}'
        if max_longitude:
            request_url += f'&maxlongitude={max_longitude}'
        if min_magnitude and delta_m:
            request_url += f'&minmagnitude={min_magnitude - (delta_m / 2)}'
        elif min_magnitude:
            request_url += f'&minmagnitude={min_magnitude}'
        if include_all_magnitudes:
            request_url += f'&includeallmagnitudes={include_all_magnitudes}'
        if event_type:
            request_url += f'&eventtype={event_type}'

        catalog = []

        parser = make_parser()
        parser.setFeature(handler.feature_namespaces, False)
        parser.setContentHandler(QuakeMLHandler(
            catalog, includeallmagnitudes=include_all_magnitudes))

        r = requests.get(request_url, stream=True)
        r.raw.decode_content = True  # if content-encoding is used decode
        parser.parse(r.raw)

        df = Catalog.from_dict(catalog)

        if not include_uncertainty:
            rgx = "(_uncertainty|_lowerUncertainty|" \
                "_upperUncertainty|_confidenceLevel)$"

            cols = df.filter(regex=rgx).columns
            df = df.drop(columns=cols)

        return df
