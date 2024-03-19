from __future__ import annotations

import os
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd

from seismostats.analysis.estimate_beta import estimate_b
from seismostats.analysis.estimate_mc import mc_ks
from seismostats.io.parser import parse_quakeml, parse_quakeml_file
from seismostats.utils import (_check_required_cols, _render_template,
                               require_cols)
from seismostats.utils.binning import bin_to_precision

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
        args : optional
            Additional arguments to pass to pandas
            DataFrame constructor.
        kwargs: optional
            Additional keyword arguments to pass to pandas
            DataFrame constructor.
    Notes:
        The Catalog class is a subclass of pandas DataFrame, and inherits
        all of its methods and attributes.

    Examples:
        Create a Catalog from a dictionary.

        >>> import pandas as pd
        >>> from seismostats.seismicity import Catalog
        >>> data = {'longitude': [0, 1, 2],
        ...         'latitude': [0, 1, 2],
        ...         'depth': [0, 1, 2],
        ...         'time': pd.to_datetime(['2021-01-01 00:00:00',
        ...                                 '2021-01-01 00:00:00',
        ...                                 '2021-01-01 00:00:00']),
        ...         'magnitude': [1, 2, 3]}
        >>> catalog = Catalog(data)
        >>> catalog
           longitude  latitude  depth                time  magnitude
        0          0         0      0 2021-01-01 00:00:00          1
        1          1         1      1 2021-01-01 00:00:00          2
        2          2         2      2 2021-01-01 00:00:00          3
    """

    _metadata = ['name', '_required_cols', 'mc',
                 'delta_m', 'b_value', 'starttime', 'endtime']
    _required_cols = REQUIRED_COLS_CATALOG

    def __init__(
        self,
        data=None,
        *args,
        name=None,
        starttime=None,
        endtime=None,
        mc=None,
        delta_m=None,
        b_value=None,
        **kwargs
    ):
        if data is None and 'columns' not in kwargs:
            super().__init__(columns=REQUIRED_COLS_CATALOG, *args,
                             starttime=None, endtime=None, **kwargs)
        else:
            super().__init__(data, *args, **kwargs)

        if self.columns.empty:
            self = self.reindex(self.columns.union(
                REQUIRED_COLS_CATALOG), axis=1)

        self.name = name
        self.mc = mc
        self.b_value = b_value
        self.delta_m = delta_m

        self.starttime = starttime if isinstance(
            starttime, pd.Timestamp) else pd.to_datetime(starttime)

        self.endtime = endtime if isinstance(
            endtime, pd.Timestamp) else pd.to_datetime(endtime)

    @classmethod
    def from_quakeml(cls, quakeml: str,
                     include_all_magnitudes: bool = True,
                     includeuncertainties: bool = False,
                     includeids: bool = False,
                     include_quality: bool = False) -> Catalog:
        """
        Create a Catalog from a QuakeML file.

        Args:
            quakeml : str
                Path to a QuakeML file or QuakeML as a string.

        Returns:
            Catalog
        """
        if os.path.isfile(quakeml):
            catalog = parse_quakeml_file(
                quakeml, include_all_magnitudes, include_quality)
        else:
            catalog = parse_quakeml(
                quakeml, include_all_magnitudes, include_quality)

        df = cls.from_dict(catalog, includeuncertainties, includeids)

        return df

    @classmethod
    def from_dict(cls,
                  data: list[dict],
                  includeuncertainty: bool = True,
                  includeids: bool = True, *args, **kwargs) -> Catalog:
        """
        Create a Catalog from a list of dictionaries.

        Args:
            data : list[dict]
                A list of earthquake event information dictionaries.

        Returns:
            Catalog
        """

        df = super().from_dict(data, *args, **kwargs)
        df = cls(df)

        numeric_cols = ['magnitude', 'latitude', 'longitude', 'depth',
                        'associatedphasecount', 'usedphasecount',
                        'associatedstationcount', 'usedstationcount',
                        'standarderror', 'azimuthalgap',
                        'secondaryazimuthalgap', 'maximumdistance',
                        'minimumdistance', 'mediandistance']

        for num in numeric_cols:
            if num in df.columns:
                df[num] = pd.to_numeric(df[num])

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

        if not includeuncertainty and isinstance(df, Catalog):
            df = df.drop_uncertainties()
        if not includeids and isinstance(df, Catalog):
            df = df.drop_ids()

        if not isinstance(df, Catalog):
            df = Catalog(df)

        if df.empty:
            df = Catalog(columns=REQUIRED_COLS_CATALOG)

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
        df.delta_m = delta_m

        if not inplace:
            return df

    @require_cols(require=['magnitude'])
    def estimate_mc(
        self,
        mcs_test: list | None = None,
        delta_m: float | None = None,
        p_pass: float = 0.05,
        stop_when_passed: bool = True,
        verbose: bool = False,
        beta: float | None = None,
        n_samples: int = 10000
    ) -> tuple[np.ndarray, list[float], np.ndarray, float | None, float | None]:
        """
        Estimate the completeness magnitude (mc), possible mc values given as
        an argument, or set to a range of values between min and max magnitude
        in the catalog.

        Args:
            mcs_test:           Completeness magnitudes to test
            delta_m:            Magnitude bins (sample has to be rounded to bins
                            beforehand)
            p_pass:             P-value with which the test is passed
            stop_when_passed:   Stop calculations when first mc passes the test,
                            by default True
            verbose:            Verbose output, by default False
            beta:               If beta is 'known', only estimate mc,
                            by default None
            n_samples:          Number of magnitude samples to be generated in
                            p-value calculation of KS distance, default 10000

        Returns:
            mcs_test:   tested completeness magnitudes
            ks_ds:      KS distances
            ps:         p-values
            best_mc:    best mc
            beta:       corresponding best beta
        """
        if delta_m is None and self.delta_m is None:
            raise ValueError("binning (delta_m) needs to be set")
        if delta_m is None:
            delta_m = self.delta_m

        if mcs_test is None:
            mcs_test = np.arange(self.magnitude.min(),
                                 self.magnitude.max(),
                                 delta_m)

        mc_est = mc_ks(self.magnitude,
                       mcs_test,
                       delta_m,
                       p_pass,
                       stop_when_passed,
                       verbose,
                       beta,
                       n_samples)

        self.mc = mc_est[3]
        return mc_est

    @require_cols(require=['magnitude'])
    def estimate_b(
        self,
        mc: float | None = None,
        delta_m: float | None = None,
        weights: list | None = None,
        b_parameter: str = "b_value",
        return_std: bool = False,
        method: str = "tinti",
        return_n: bool = False,
    ) -> float | tuple[float, float] | tuple[float, float, float]:
        """
        Estimates b-value of magnitudes in the Catalog based on settings given
        by the input parameters. Sets attribute b-value to the computed value,
        but also returns the computed b-value. If return_std or return_n set
        to True, also returns the uncertainty and/or number of magnitudes used
        to estimate the b-value.

        Args:
            mc:         completeness magnitude, etiher given as parameter or
                    taken from the object attribute
            delta_m:    discretization of magnitudes, etiher given as parameter
                    or taken from the object attribute
            weights:    weights of each magnitude can be specified here
            b_parameter:either 'b-value', then the corresponding value  of the
                    Gutenberg-Richter law is returned, otherwise 'beta'
                    from the exponential distribution [p(M) = exp(-beta*(M-mc))]
            return_std: if True the standard deviation of beta/b-value (see
                    above) is returned
            method:     method to use for estimation of beta/b-value. Options
                    are: 'tinti', 'utsu', 'positive', 'laplace'
            return_n:   if True, the number of events used for the estimation is
                    returned. This is only relevant for the 'positive' method

        Returns:
            b:      maximum likelihood beta or b-value, depending on value of
                input variable 'b_parameter'. Note that the difference
                is just a factor [b_value = beta * log10(e)]
            std:    Shi and Bolt estimate of the beta/b-value error estimate
            n:      number of events used for the estimation
        """

        if mc is None and self.mc is None:
            raise ValueError("completeness magnitude (mc) needs to be set")
        if mc is None:
            mc = self.mc

        if delta_m is None and self.delta_m is None:
            raise ValueError("binning (delta_m) needs to be set")
        if delta_m is None:
            delta_m = self.delta_m

        b_estimate = estimate_b(self.magnitude,
                                mc,
                                delta_m,
                                weights,
                                b_parameter,
                                return_std,
                                method,
                                return_n)

        if return_std or return_n:
            self.b_value = b_estimate[0]
        else:
            self.b_value = b_estimate

        return b_estimate

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

    @require_cols(require=_required_cols + ['magnitude_type'])
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

    def __finalize__(self, other, method=None, **kwargs):
        """ propagate metadata from other to self
            Source: https://github.com/geopandas/geopandas
        """
        self = super().__finalize__(other, method=method, **kwargs)

        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(
                    other.objs[0], name, None))
        return self


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

    def __init__(self, data=None, *args, name=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        # Represent catalogs with no events
        self.empty_catalogs = None

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID=' ', author=' ') -> str:
        catalogs = []
        for _, group in self.groupby('catalog_id'):
            catalogs.append(Catalog(group).to_quakeml(agencyID, author))
        return catalogs
