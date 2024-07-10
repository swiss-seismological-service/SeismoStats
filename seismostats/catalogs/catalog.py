from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Any

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
        data:       array-like, Iterable, dict, or DataFrame, optional
                    Data to initialize the catalog with.
        name:       Name of the catalog.
        args:       Additional arguments to pass to pandas
                    DataFrame constructor.
        starttime:  Start time of the catalog.
        endtime:    End time of the catalog.
        mc:         Completeness magnitude of the catalog.
        delta_m:    Magnitude binning of the catalog.
        kwargs:     Additional keyword arguments to pass to pandas
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
        data: Any | None = None,
        *args,
        name: str | None = None,
        starttime: pd.Timestamp | None = None,
        endtime: pd.Timestamp | None = None,
        mc: float | None = None,
        delta_m: float | None = None,
        b_value: float | None = None,
        **kwargs
    ):
        if data is None and 'columns' not in kwargs:
            super().__init__(columns=REQUIRED_COLS_CATALOG, *args, **kwargs)
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
                     include_uncertainties: bool = False,
                     include_ids: bool = False,
                     include_quality: bool = False) -> Catalog:
        """
        Create a Catalog from a QuakeML file.

        Args:
            quakeml:                Path to a QuakeML file or QuakeML
                                    as a string.
            include_all_magnitudes: Whether all available magnitude types
                                    should be included.
            include_uncertainties:  Whether value columns with uncertainties
                                    should be included.
            include_ids:            Whether event, origin, and magnitude IDs
                                    should be included.
            include_quality:        Whether columns with quality information
                                    should be included.

        Returns:
            Catalog
        """
        if os.path.isfile(quakeml):
            catalog = parse_quakeml_file(
                quakeml, include_all_magnitudes, include_quality)
        else:
            catalog = parse_quakeml(
                quakeml, include_all_magnitudes, include_quality)

        df = cls.from_dict(catalog, include_uncertainties, include_ids)

        return df

    @classmethod
    def from_dict(cls,
                  data: list[dict],
                  include_uncertainties: bool = True,
                  include_ids: bool = True, *args, **kwargs) -> Catalog:
        """
        Create a Catalog from a list of dictionaries.

        Args:
            data:                   A list of earthquake event information
                                    dictionaries.
            include_uncertainties:  Whether value columns with uncertainties
                                    should be included.
            include_ids:            Whether event, origin, and magnitude IDs
                                    should be included.

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

        if not include_uncertainties and isinstance(df, Catalog):
            df = df.drop_uncertainties()
        if not include_ids and isinstance(df, Catalog):
            df = df.drop_ids()

        if not isinstance(df, Catalog):
            df = Catalog(df)

        if df.empty:
            df = Catalog(columns=REQUIRED_COLS_CATALOG)

        return df

    def drop_uncertainties(self) -> Catalog:
        """
        Drop uncertainty columns from the catalog.

        Drops columns with names ending in '_uncertainty', '_lowerUncertainty',
        '_upperUncertainty', and '_confidenceLevel'.

        Returns:
            catalog: Catalog with uncertainty columns removed.
        """

        rgx = "(_uncertainty|_lowerUncertainty|" \
            "_upperUncertainty|_confidenceLevel)$"

        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

    def drop_ids(self) -> Catalog:
        """
        Drop event, origin, and magnitude IDs from the catalog.

        Drops columns named 'eventid', 'originid', and 'magnitudeid'.

        Returns:
            catalog: Catalog with ID columns removed.
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
        Remove all columns except the required ones
        defined in ``_required_cols``.

        Args:
            inplace:    Whether to perform the operation in place on the data.

        Returns:
            catalog:    Catalog with the stripped columns.
        """
        df = self.drop(columns=set(self.columns).difference(
            set(self._required_cols)), inplace=inplace)
        if not inplace:
            return df

    @require_cols(require=['magnitude'])
    def bin_magnitudes(self, delta_m: float = 0.1, inplace: bool = False) \
            -> Catalog | None:
        """
        Rounds values in the ``magnitude`` column of the catalog to a given
        precision ``delta_m``.

        Args:
            delta_m:    size of the bin, optional
            inplace:    Whether to perform the operation in place on the data.

        Returns:
            catalog:    Catalog with rounded magnitudes.
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
            mcs_test:           Completeness magnitudes to test.
            delta_m:            Magnitude bins (sample has to be rounded to bins
                                beforehand).
            p_pass:             P-value with which the test is passed.
            stop_when_passed:   Stop calculations when first mc passes the test.
            verbose:            Verbose output.
            beta:               If beta is 'known', only estimate mc.
            n_samples:          Number of magnitude samples to be generated in
                                p-value calculation of KS distance.

        Returns:
            mcs_test:   Tested completeness magnitudes.
            ks_ds:      KS distances.
            ps:         p-values.
            best_mc:    Best mc.
            beta:       Corresponding best beta.
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
            mc:         Completeness magnitude, etiher given as parameter or
                        taken from the object attribute.
            delta_m:    Discretization of magnitudes, etiher given as parameter
                        or taken from the object attribute.
            weights:    Weights of each magnitude can be specified here.
            b_parameter:Either 'b-value', then the corresponding value  of the
                        Gutenberg-Richter law is returned, otherwise 'beta'
                        from the exponential distribution
                        :math:`p(M) = exp(-beta*(M-mc))`
            return_std: If True the standard deviation of beta/b-value (see
                        above) is returned.
            method:     Method to use for estimation of beta/b-value. Options
                        are: 'tinti', 'utsu', 'positive', 'laplace'.
            return_n:   If True, the number of events used for the estimation is
                        returned. This is only relevant for the 'positive'
                        method.

        Returns:
            b:      Maximum likelihood beta or b-value, depending on value of
                    input variable 'b_parameter'. Note that the difference
                    is just a factor :math:`b = beta * log10(e)`.
            std:    Shi and Bolt estimate of the beta/b-value error estimate
            n:      number of events used for the estimation.
        """

        if mc is None and self.mc is None:
            raise ValueError("completeness magnitude (mc) needs to be set")
        if mc is None:
            mc = self.mc

        if delta_m is None and self.delta_m is None:
            raise ValueError("binning (delta_m) needs to be set")
        if delta_m is None:
            delta_m = self.delta_m

        if method == "positive":
            # dataframe needs 'time' column to be sorted
            if 'time' not in self.columns:
                raise ValueError('"time" column needs to be set in order to \
                                 use the b-positive method')
            mags = self.sort_values("time").magnitude
        else:
            mags = self.magnitude

        b_estimate = estimate_b(mags,
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

        Returns:
            keys: List of secondary magnitude keys in the catalog.
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

    def _create_ids(self) -> Catalog:
        """
        Create missing event, origin, and magnitude IDs for the catalog.

        Will fill in missing IDs with UUIDs.
        """
        df = self.copy()

        if 'eventid' not in df.columns:
            df['eventid'] = uuid.uuid4()
        if 'originid' not in df.columns:
            df['originid'] = uuid.uuid4()
        if 'magnitudeid' not in df.columns:
            df['magnitudeid'] = uuid.uuid4()

        mag_types = set([mag.split('_')[1]
                        for mag in df._secondary_magnitudekeys()])

        for mag_type in mag_types:
            if f'magnitude_{mag_type}_magnitudeid' not in df.columns:
                df[f'magnitude_{mag_type}_magnitudeid'] = \
                    df.apply(
                        lambda x: uuid.uuid4()
                        if not mag_type == x['magnitude_type']
                        else x['magnitudeid'], axis=1)

        return df

    @require_cols(require=_required_cols + ['magnitude_type'])
    def to_quakeml(self, agencyID=' ', author=' ') -> str:
        """
        Convert the catalog to QuakeML format.

        Args:
            agencyID:   Agency ID with which to store the catalog.
            author:     Author of the catalog.

        Returns:
            catalog:    The catalog in QuakeML format.
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

    def __finalize__(self, other, method=None, **kwargs) -> Catalog:
        """ Propagate metadata from other to self.
            Source: https://github.com/geopandas/geopandas

        Args:
            other:  The other object to finalize with.
            method: The method used to finalize the objects.
            kwargs: Additional keyword arguments.

        Returns:
            self:   The finalized object.
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
        data: array-like, Iterable, dict, or DataFrame, optional.
                    Data to initialize the catalog with.
        name:       Name of the catalog.
        n_catalogs: Total number of catalogs represented,
                    including empty catalogs.
        args:       Additional arguments to pass to pandas
                    DataFrame constructor.
        starttime:  Start time of the catalog.
        endtime:    End time of the catalog.
        mc:         Completeness magnitude of the catalog.
        delta_m:    Magnitude binning of the catalog.
        kwargs:     Additional keyword arguments to pass to pandas
                    DataFrame constructor.

    Notes:
        The Catalog class is a subclass of pandas DataFrame, and inherits
        all of its methods and attributes.
    """

    _required_cols = REQUIRED_COLS_CATALOG + ['catalog_id']
    _metadata = Catalog._metadata + ['n_catalogs']

    def __init__(self, data=None, *args, n_catalogs=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        # Total number of catalogs represented, inculding empty catalogs
        self.n_catalogs = n_catalogs

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID: str = ' ', author: str = ' ') -> list[str]:
        """
        Convert the catalogs to QuakeML format.

        Args:
            agencyID:   Agency ID.
            author:     Author of the catalog.

        Returns:
            catalogs:   List of catalogs in QuakeML format.
        """
        catalogs = []
        for _, group in self.groupby('catalog_id'):
            catalogs.append(Catalog(group).to_quakeml(agencyID, author))
        return catalogs
