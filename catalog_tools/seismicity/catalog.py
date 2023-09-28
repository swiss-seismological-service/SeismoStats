from __future__ import annotations

import os
import uuid
from collections import defaultdict

import pandas as pd

from catalog_tools.io.parser import parse_quakeml, parse_quakeml_file
from catalog_tools.utils import (_check_required_cols, _render_template,
                                 require_cols)
from catalog_tools.utils.binning import bin_to_precision

REQUIRED_COLS_CATALOG = ['longitude', 'latitude', 'depth',
                         'time', 'magnitude']

QML_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'catalog_templates', 'quakeml.j2')


def _catalog_constructor_with_fallback(*args, **kwargs):
    df = Catalog(*args, **kwargs)
    if not _check_required_cols(df, REQUIRED_COLS_CATALOG):
        return pd.DataFrame(*args, **kwargs)
    if not _check_required_cols(df, required_cols=['catalog_id']):
        return df
    return ForecastCatalog(*args, **kwargs)


class Catalog(pd.DataFrame):
    """
    A subclass of pandas DataFrame that represents a catalog of earthquakes.

    To be a valid Catalog object, the DataFrame must have the following
    columns: longitude, latitude, depth, time, and magnitude.

    Args:
        data : array-like, Iterable, dict, or DataFrame, optional
            Data to initialize the catalog with.
        name : str, optional
            Name of the catalog.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to pandas
            DataFrame constructor.

    Notes:
        The Catalog class is a subclass of pandas DataFrame, and inherits
        all of its methods and attributes.
    """
    _metadata = ['name', '_required_cols']
    _required_cols = REQUIRED_COLS_CATALOG

    def __init__(self, data=None, *args, name=None, **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

    @classmethod
    def from_quakeml(cls, quakeml: str,
                     includeallmagnitudes: bool = True,
                     includeuncertainties: bool = False,
                     includeids: bool = False) -> Catalog:
        """
        Create a Catalog from a QuakeML file.

        Args:
            quakeml : str
                Path to a QuakeML file or QuakeML as a string.

        Returns:
            Catalog
        """
        if os.path.isfile(quakeml):
            catalog = parse_quakeml_file(quakeml, includeallmagnitudes)
        else:
            catalog = parse_quakeml(quakeml, includeallmagnitudes)

        df = cls.from_dict(catalog)

        if not includeuncertainties:
            df = df.drop_uncertainties()
        if not includeids:
            df = df.drop_ids()

        return df

    def drop_uncertainties(self):
        """
        Drop uncertainty columns from the catalog.
        """

        rgx = "(_uncertainty|_lowerUncertainty|" \
            "_upperUncertainty|_confidenceLevel)$"

        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

    def drop_ids(self):
        """
        Drop event, origin, and magnitude IDs from the catalog.
        """

        rgx = "(eventid|originid|magnitudeid)$"
        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

    @property
    def _constructor(self):
        return _catalog_constructor_with_fallback

    @require_cols(require=_required_cols)
    def strip(self, inplace: bool = False) -> Catalog | None:
        """
        Remove all columns except the required ones.

        Args:
            inplace : bool, optional
                If True, do operation inplace.

        Returns:
            Catalog or None
                If inplace is True, returns None. Otherwise, returns a new
                Catalog with the stripped columns.
        """
        df = self.drop(columns=set(self.columns).difference(
            set(self._required_cols)), inplace=inplace)
        if not inplace:
            return df

    @require_cols(require=['magnitude'])
    def bin_magnitudes(
            self, delta_m: float, inplace: bool = False) -> Catalog | None:
        """
        Bin magnitudes to a given precision.

        Args:
            delta_m : float
                Magnitude bin size.
            inplace : bool, optional
                If True, do operation inplace.

        Returns:
            Catalog or None
        """
        if inplace:
            df = self
        else:
            df = self.copy()

        df['magnitude'] = bin_to_precision(df["magnitude"], delta_m)

        if not inplace:
            return df

    def _secondary_magnitudekeys(self) -> list[str]:
        """
        Get a list of secondary magnitude keys in the catalog.

        This will always include also the preferred magnitude type.
        """

        vals = ['_uncertainty',
                '_lowerUncertainty',
                '_upperUncertainty',
                '_confidenceLevel',
                '_type']

        secondary_mags = [mag for mag in self.columns if
                          'magnitude_' in mag
                          and not any(['magnitude' + val
                                       in mag for val in vals])]
        return secondary_mags

    def _create_ids(self):
        """
        Create missing event, origin, and magnitude IDs for the catalog.

        Will fill in missing IDs with UUIDs.
        """

        if 'eventid' not in self.columns:
            self['eventid'] = uuid.uuid4()
        if 'originid' not in self.columns:
            self['originid'] = uuid.uuid4()
        if 'magnitudeid' not in self.columns:
            self['magnitudeid'] = uuid.uuid4()

        mag_types = set([mag.split('_')[1]
                        for mag in self._secondary_magnitudekeys()])

        for mag_type in mag_types:
            if f'magnitude_{mag_type}_magnitudeid' not in self.columns:
                self[f'magnitude_{mag_type}_magnitudeid'] = \
                    self.apply(
                        lambda x: uuid.uuid4()
                        if not mag_type == x['magnitude_type']
                        else x['magnitudeid'], axis=1)

        return self

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID=' ', author=' ') -> str:
        """
        Convert the catalog to QuakeML format.

        Args:
            agencyID : str, optional
                Agency ID.
            author : str, optional
                Author of the catalog.

        Returns:
            str
                The catalog in QuakeML format.
        """

        df = self.copy()
        df = df._create_ids()

        secondary_mags = self._secondary_magnitudekeys()

        data = dict(events=df.to_dict(orient='records'),
                    agencyID=agencyID, author=author)

        for event in data['events']:
            event['sec_mags'] = defaultdict(dict)
            for mag in secondary_mags:
                if pd.notna(event[mag]) \
                        and event['magnitude_type'] not in mag:

                    mag_type = mag.split('_')[1]
                    mag_key = mag.replace('_' + mag_type, '')

                    event['sec_mags'][mag_type][mag_key] = \
                        event[mag]
                del event[mag]

        return _render_template(data, QML_TEMPLATE)


class ForecastCatalog(Catalog):
    """
    A subclass of pandas DataFrame that represents catalogs of earthquake
    forecasts.

    To be a valid ForecastCatalog object, the DataFrame must have the
    following columns: longitude, latitude, depth, time, magnitude,
    catalog_id.

    Args:
        data : array-like, Iterable, dict, or DataFrame, optional
            Data to initialize the catalog with.
        name : str, optional
            Name of the catalog.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to pandas
            DataFrame constructor.

    Notes:
        The ForecastCatalog class is a subclass of pandas DataFrame, and
        inherits all of its methods and attributes.
    """

    _required_cols = REQUIRED_COLS_CATALOG + ['catalog_id']

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID=' ', author=' ') -> str:
        catalogs = []
        for _, group in self.groupby('catalog_id'):
            catalogs.append(Catalog(group).to_quakeml(agencyID, author))
        return catalogs
