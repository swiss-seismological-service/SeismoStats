from __future__ import annotations

import inspect
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import Polygon

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.estimate_mc import mc_ks
from seismostats.io.parser import parse_quakeml, parse_quakeml_file
from seismostats.plots.basics import (plot_cum_count, plot_cum_fmd, plot_fmd,
                                      plot_mags_in_time)
from seismostats.plots.seismicity import plot_in_space
from seismostats.plots.statistical import plot_mc_vs_b
from seismostats.utils import (_check_required_cols, _render_template,
                               require_cols)
from seismostats.utils.binning import bin_to_precision

try:
    from openquake.hmtk.seismicity.catalogue import Catalogue as OQCatalogue
except ImportError:  # pragma: no cover
    _openquake_available = False
else:
    _openquake_available = True

CATALOG_COLUMNS = ['longitude', 'latitude', 'depth',
                   'time', 'magnitude', 'magnitude_type']

QML_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'catalog_templates', 'quakeml.j2')

_PD_TIME_COLS = ['year', 'month', 'day',
                 'hour', 'minute', 'second', 'microsecond']


def _catalog_constructor_with_fallback(df, **kwargs):
    if not _check_required_cols(df, ['magnitude']):
        return df
    if not _check_required_cols(df, ['catalog_id']):
        return Catalog(df, **kwargs)
    return ForecastCatalog(df, **kwargs)


class Catalog(pd.DataFrame):
    """
    A subclass of pandas DataFrame that represents a catalog of earthquakes.

    To be a valid Catalog object, the DataFrame must have at least a
    `magnitude` column. Depending on the method the following
    columns are also required: `longitude`, `latitude`, `depth`, `time`,
    and `magnitude` .

    Args:
        data:       Data to initialize the catalog with.
        name:       Name of the catalog.
        starttime:  Start time of the catalog.
        endtime:    End time of the catalog.
        mc:         Completeness magnitude of the catalog.
        delta_m:    Magnitude binning of the catalog.
        b_value:    Gutenberg-Richter b-value of the catalog.
        bounding_polygon: 2D boundary of the catalog.
        depth_min:  Minimum depth for which events are included in the catalog.
        depth_max:  Maximum depth for which events are included in the catalog.
        kwargs:     Additional keyword arguments to pass to pandas
                DataFrame constructor.

    See Also:
        The Catalog class is a subclass of :class:`pandas.DataFrame`, and
        inherits all of its methods and attributes.

    Examples:
        Create a Catalog from a dictionary.
        .. code-block:: python

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

        :ivar name:         Name of the catalog.
        :ivar mc:           Completeness magnitude of the catalog.
        :ivar delta_m:      Magnitude binning of the catalog.
        :ivar b_value:      Gutenberg-Richter b-value of the catalog.
        :ivar starttime:    Start time of the catalog.
        :ivar endtime:      End time of the catalog.
        :ivar bounding_polygon: 2D boundary of the catalog.
        :ivar depth_min:    Minimum depth for which events are included in the
                        catalog.
        :ivar depth_max:    Maximum depth for which events are included in the
                        catalog.
        :ivar logger:       Logger for the catalog.
    """

    _metadata = ['name', '_required_cols', 'mc',
                 'delta_m', 'b_value', 'starttime', 'endtime',
                 'bounding_polygon', 'depth_min', 'depth_max',
                 'logger']

    _required_cols = CATALOG_COLUMNS

    @property
    def _constructor(self):
        """
        Required for subclassing Pandas DataFrame.
        """
        return _catalog_constructor_with_fallback

    def __init__(
            self,
            data: Any | None = None,
            name: str | None = None,
            starttime: pd.Timestamp | None = None,
            endtime: pd.Timestamp | None = None,
            mc: float | None = None,
            delta_m: float | None = None,
            b_value: float | None = None,
            bounding_polygon: Polygon | str | None = None,
            depth_min: float | None = None,
            depth_max: float | None = None,
            **kwargs):

        self.logger = logging.getLogger(__name__)

        # should be able to create a dataframe
        if data is not None or 'columns' in kwargs:
            super().__init__(data, **kwargs)
        # if this dataframe is empty however, set some default columns
        if data is None or self.columns.empty:
            super().__init__(columns=CATALOG_COLUMNS, **kwargs)

        self.name = name
        self.mc = mc
        self.b_value = b_value
        self.delta_m = delta_m
        self.starttime = pd.to_datetime(starttime)
        self.endtime = pd.to_datetime(endtime)
        self.bounding_polygon = bounding_polygon
        self.depth_min = depth_min
        self.depth_max = depth_max

        numeric_cols = ['magnitude', 'latitude', 'longitude', 'depth',
                        'associatedphasecount', 'usedphasecount',
                        'associatedstationcount', 'usedstationcount',
                        'standarderror', 'azimuthalgap',
                        'secondaryazimuthalgap', 'maximumdistance',
                        'minimumdistance', 'mediandistance']
        string_cols = ['magnitude_type', 'event_type']
        time_cols = ['time']

        for num in numeric_cols:
            if num in self.columns:
                self[num] = pd.to_numeric(self[num], errors='coerce')

        for tc in time_cols:
            if tc in self.columns:
                self[tc] = pd.to_datetime(self[tc]).dt.tz_localize(None)

        # make sure empty rows in string columns are NoneType
        for strc in string_cols:
            if strc in self.columns:
                self[strc] = self[strc].replace(
                    to_replace=['',
                                'nan', 'NaN',
                                'none', 'None',
                                'na', 'Na', 'NA',
                                'null', 'Null', 'NULL'],
                    value=None)

    @classmethod
    def from_quakeml(cls,
                     quakeml: str,
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
            catalog:                Catalog object

        Examples:
            .. code-block:: python

                >>> from seismostats import Catalog
                >>> catalog = Catalog.from_quakeml("path/to/quakeml.xml")
                >>> catalog
                   longitude   latitude  depth                time  magnitude
                0      42.35    3.34444   5.50 1900-01-01 05:05:13        1.0
                1       1.35    5.13500  10.52 1982-04-07 07:07:15        2.5
                2       2.35    2.13400  50.40 2020-11-30 12:30:59        3.9
        """
        if os.path.isfile(quakeml):
            catalog = parse_quakeml_file(
                quakeml, include_all_magnitudes, include_quality)
        else:
            catalog = parse_quakeml(
                quakeml, include_all_magnitudes, include_quality)

        df = cls.from_dict(catalog,
                           include_uncertainties,
                           include_ids)

        return df

    @classmethod
    def from_dict(cls,
                  data: list[dict],
                  include_uncertainties: bool = True,
                  include_ids: bool = True,
                  **kwargs) -> Catalog:
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

        Examples:
            .. code-block:: python

                >>> from seismostats import Catalog
                >>> data = [
                ...     {'longitude': 42.35, 'latitude': 3.34444,
                ...      'depth': 5.50, 'time': '1900-01-01 05:05:13',
                ...      'magnitude': 1.0},
                ...     {'longitude': 1.35, 'latitude': 5.13500,
                ...      'depth': 10.52, 'time': '1982-04-07 07:07:15',
                ...      'magnitude': 2.5}]
                >>> catalog = Catalog.from_dict(data)
                >>> catalog
                   longitude   latitude  depth                time  magnitude
                0      42.35    3.34444   5.50 1900-01-01 05:05:13        1.0
                1       1.35    5.13500  10.52 1982-04-07 07:07:15        2.5
        """

        df = pd.DataFrame.from_dict(data, **kwargs)
        df = cls(df)

        if not include_uncertainties:  # and isinstance(df, Catalog):
            df = df.drop_uncertainties()
        if not include_ids:  # and isinstance(df, Catalog):
            df = df.drop_ids()

        return df

    @classmethod
    def from_openquake(cls, oq_catalogue: OQCatalogue,
                       keep_time_cols=False) -> Catalog:
        """
        Create a (seismostats) Catalog from an openquake Catalogue.
        The optional dependency group openquake is required for this method.

        Args:
            oq_catalogue:       The openquake catalogue.
            keep_time_cols:     Whether the time columns: 'year', 'month',
                            'day', 'hour', 'minute', 'second'
                            should be kept (they are converted to 'time').
        Returns:
            Catalog

        Examples:
            .. code-block:: python

                >>> from openquake.hmtk.seismicity.catalogue import \
                ...     Catalogue as OQCatalog
                >>> from seismostats import Catalog

                >>> simple_oq_catalogue = OQCatalog.make_from_dict({
                ...     'eventID': ["event0", "event1", "event2"],
                ...     'longitude': np.array([42.35, 1.35, 2.35], dtype=float),
                ...     'latitude': np.array([3.34444, 5.135, 2.134],
                ...                          dtype=float),
                ...     'depth': np.array([5.5, 10.52, 50.4], dtype=float),
                ...     'year': np.array([1900, 1982, 2020], dtype=int),
                ...     'month': np.array([1, 4, 11], dtype=int),
                ...     'day': np.array([1, 7, 30], dtype=int),
                ...     'hour': np.array([5, 7, 12], dtype=int),
                ...     'minute': np.array([5, 7, 30], dtype=int),
                ...     'second': np.array([13.1234, 15.0, 59.9999],
                ...                        dtype=float),
                ...     'magnitude': np.array([1.0, 2.5, 3.9], dtype=float)
                ...     })
                >>> catalog = Catalog.from_openquake(simple_oq_catalogue)
                >>> catalog
                   longitude   latitude  depth                time  magnitude
                0      42.35    3.34444   5.50 1900-01-01 05:05:13        1.0
                1       1.35    5.13500  10.52 1982-04-07 07:07:15        2.5
                2       2.35    2.13400  50.40 2020-11-30 12:30:59        3.9
        """
        if not _openquake_available:
            raise ImportError("the optional openquake package is not available")

        def _convert_to_datetime(row):
            return datetime(row.year,
                            row.month,
                            row.day,
                            row.hour,
                            row.minute,
                            row.second)
        data = oq_catalogue.data
        # not all items of the data have necessarily the same length
        length = max((len(c) for c in data.values()), default=0)

        if length == 0:
            return cls()

        cat = cls({k: v for k, v in data.items() if len(v) > 0})
        # hmtk stores seconds as floats, but pandas requires them as integers
        us = ((cat["second"] % 1) * 1e6)
        cat["microsecond"] = us.round().astype(np.int32)
        cat["second"] = cat["second"].astype(np.int32)
        try:
            cat.loc[:, "time"] = pd.to_datetime(cat[_PD_TIME_COLS])
        except ValueError:
            # if the time is out of bounds, we have to store
            # datetime with a resolution of seconds.
            dt = cat.apply(_convert_to_datetime, axis=1)
            cat['time'] = dt.astype('datetime64[s]')
        if not keep_time_cols:
            cat.drop(columns=_PD_TIME_COLS, inplace=True)
        return cat

    @require_cols(require=[
        'longitude', 'latitude', 'depth', 'time', 'magnitude'])
    def to_openquake(self) -> OQCatalogue:
        """
        Converts the Catalog to an openquake Catalogue
        The optional dependency group openquake is required for this method.
        The required columns are mapped to the openquake columns, except
        time is converted to 'year', 'month', 'day', 'hour', 'minute', 'second'.
        'eventID' is created if not present.

        Returns:
            OQCatalogue:        the converted Catalogue

        Examples:
            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'depth': [5.5, 10.52, 50.4],
                ...     'time': pd.to_datetime(['1900-01-01 05:05:13',
                ...                             '1982-04-07 07:07:15',
                ...                             '2020-11-30 12:30:59']),
                ...     'magnitude': [1.0, 2.5, 3.9]
                ...     })
                >>> oq_catalog = simple_catalog.to_openquake()
                >>> type(oq_catalog)
                <class 'openquake.hmtk.seismicity.catalogue.Catalogue'>
        """
        if not _openquake_available:
            raise ImportError("the optional openquake package is not available")
        if len(self) == 0:
            return OQCatalogue()
        data = dict()
        for col, dtype in zip(self.columns, self.dtypes):
            if np.issubdtype(dtype, np.number):
                data[col] = self[col].to_numpy(dtype=dtype, copy=True)
            else:
                data[col] = self[col].to_list()
        # add required eventID if not present
        if 'eventID' not in data:
            data['eventID'] = self.apply(
                lambda _: uuid.uuid4().hex, axis=1).to_list()

        time = self['time']
        for time_unit in _PD_TIME_COLS:
            data[time_unit] = getattr(
                time.dt, time_unit).to_numpy(copy=True)
        data["second"] = data["second"] + data["microsecond"] / 1e6
        return OQCatalogue.make_from_dict(data)

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

        Drops columns named 'eventID', 'originID', and 'magnitudeID'.

        Returns:
            catalog: Catalog with ID columns removed.
        """

        rgx = "(eventID|originID|magnitudeID)$"
        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

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
    def bin_magnitudes(self, delta_m: float = None, inplace: bool = False) \
            -> Catalog | None:
        """
        Rounds values in the ``magnitude`` column of the catalog to a given
        precision ``delta_m``.

        Args:
            delta_m:    The size of the bins to round the magnitudes to.
            inplace:    Whether to perform the operation in place on the data.

        Returns:
            catalog:    Catalog with rounded magnitudes.

        Examples:
            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.02, 2.53, 3.99]
                ...     })
                >>> simple_catalog.bin_magnitudes(delta_m=0.1)
                    longitude   latitude  magnitude
                0       42.35    3.34444        1.0
                1        1.35    5.13500        2.5
                2        2.35    2.13400        4.0

            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.02, 2.53, 3.99]
                ...     })
                >>> simple_catalog.delta_m = 0.1
                >>> simple_catalog.bin_magnitudes()
                    longitude   latitude  magnitude
                0       42.35    3.34444        1.0
                1        1.35    5.13500        2.5
                2        2.35    2.13400        4.0
        """
        if delta_m is None and self.delta_m is None:
            raise ValueError("binning (delta_m) needs to be set")
        if delta_m is None:
            delta_m = self.delta_m

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
        delta_m: float | None = None,
        mcs_test: list | None = None,
        p_pass: float = 0.1,
        stop_when_passed: bool = True,
        verbose: bool = False,
        b_value: float | None = None,
        b_method: BValueEstimator = ClassicBValueEstimator,
        n: int = 10000,
        ks_ds_list: list[list] | None = None,
        *args,
        **kwargs,
    ) -> tuple[float | None, float | None, list[float],
               list[float], list[float], np.ndarray]:
        """
        Returns the smallest magnitude in a given list of completeness
        magnitudes for which the KS test is passed, i.e., where the null
        hypothesis that the sample is drawn from a Gutenberg-Richter law cannot
        be rejected.

        Source:
            - Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
              distributions in empirical data. SIAM review, 51(4), pp.661-703.
            - Mizrahi, L., Nandan, S. and Wiemer, S., 2021. The effect of
              declustering on the size distribution of mainshocks. Seismological
              Society of America, 92(4), pp.2333-2342.

        Args:
            delta_m:        Bin size of discretized magnitudes. Sample has to be
                        rounded to bins beforehand).
            mcs_test:       Array of tested completeness magnitudes. If `None`,
                        it will be generated automatically based on `sample`
                        and `delta_m`.
            p_pass:         Boolean that indicates whether to stop calculations
                        when first mc passes the test.
            stop_when_passed:  Stop calculations when first mc passes the test.
            verbose:           Boolean that indicates whether to print verbose
                        output.
            b_value:        If `b_value` is 'known', only estimate `mc` assuming
                        the given `b_value`. If `None`, the b-value is either
                        taken from the object attribute or estimated.
            b_method:       b-value estimator to use if b-value needs to be
                        calculated from data.
            n:              Number of number of times the KS distance is
                        calculated for estimating the p-value.
            ks_ds_list:     KS distances from synthetic data with the given
                        parameters. If `None`, they will be estimated here.
            *args,          Additional arguments for the b-value estimator.
            **kwargs:       Additional keyword arguments for the b-value
                        estimator.

        Returns:
            best_mc:        `mc` for which the p-value is lowest.
            best_b_value:   `b_value` corresponding to the best `mc`.
            mcs_test:       Tested completeness magnitudes.
            b_values:       Tested b-values.
            ks_ds:          KS distances.
            ps:             Corresponding p-values.

        Examples:
            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.0, 2.5, 3.9]
                ...     })
                >>> simple_catalog.estimate_mc()
                (1.0, 0.28645181449530005, [1.0], [0.28645181449530005],
                    [0.29485562894395984], array([0.866]))
        """
        if delta_m is None and self.delta_m is None:
            raise ValueError("Binning (delta_m) needs to be set.")
        if delta_m is None:
            delta_m = self.delta_m

        if b_value is None and self.b_value is not None:
            b_value = self.b_value

        best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps = \
            mc_ks(self.magnitude,
                  delta_m,
                  mcs_test,
                  p_pass,
                  stop_when_passed,
                  verbose,
                  b_value,
                  b_method,
                  n,
                  ks_ds_list,
                  *args,
                  **kwargs)

        self.mc = best_mc

        return best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps

    @require_cols(require=['magnitude'])
    def estimate_b(
        self,
        mc: float | None = None,
        delta_m: float | None = None,
        weights: list | None = None,
        method: BValueEstimator = ClassicBValueEstimator,
        *args,
        **kwargs
    ) -> BValueEstimator:
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
            b_parameter: Either 'b-value', then the corresponding value  of the
                    Gutenberg-Richter law is returned, otherwise 'beta' from
                    the exponential distribution
                    :math:`p(M) = exp(-beta*(M-mc))`
            return_std: If True the standard deviation of beta/b-value (see
                        above) is returned.
            method:     Method to use for estimation of beta/b-value. Options
                        are: 'tinti', 'utsu', 'positive', 'laplace'.
            return_n:   If True, the number of events used for the estimation
                    is returned. This is only relevant for the 'positive'
                    method.

        Returns:
            b:      Maximum likelihood beta or b-value, depending on value of
                input variable 'b_parameter'. Note that the difference
                is just a factor :math:`b = beta * log10(e)`.
            std:    Shi and Bolt estimate of the beta/b-value error estimate
            n:      number of events used for the estimation.

        Examples:
            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.0, 2.5, 3.9]
                ...     })
                >>> simple_catalog.estimate_b(mc=1.0, delta_m=0.1)
                0.28645181449530005
        """
        if mc is None and self.mc is None:
            raise ValueError("Completeness magnitude (mc) needs to be set.")
        if mc is None:
            mc = self.mc

        if delta_m is None and self.delta_m is None:
            raise ValueError("Binning (delta_m) needs to be set.")
        if delta_m is None:
            delta_m = self.delta_m

        if not method._weights_supported:
            weights = None
        elif weights is not None:
            pass
        elif 'weight' in self.columns:
            weights = self.weight

        # The next part checks, whether one argument required for the
        # 'calculate' method is available as a column in the dataframe.
        # If so, it is passed to the method.

        # Get the argument names, skipping 'self'
        sig = inspect.signature(method.calculate)
        extra_args = [
            name for name, _ in sig.parameters.items()
            if name not in ['self', 'magnitudes', 'mc', 'delta_m', 'weights']]
        # singular form version of the extra arguments
        sing_args = [item if not item.endswith('s') else
                     item[:-1] for item in extra_args]

        # Collect the matching arguments from the DataFrame
        col_kwargs = {
            arg: self[sarg] for arg, sarg in zip(extra_args + extra_args,
                                                 sing_args + extra_args)
            if sarg in self.columns
        }
        col_kwargs.update(**kwargs)

        # Create and call the estimator
        b_estimator = method()
        b_estimator.calculate(self.magnitude,
                              mc,
                              delta_m,
                              *args,
                              weights=weights,
                              **col_kwargs)

        self.b_value = b_estimator.b_value

        return b_estimator

    @require_cols(require=['latitude', 'longitude', 'magnitude'])
    def plot_in_space(self,
                      resolution: str = "10m",
                      include_map: bool | None = False,
                      country: str | None = None,
                      colors: str | None = None,
                      style: str = "satellite",
                      dot_smallest: int = 10,
                      dot_largest: int = 200,
                      dot_interpolation_power: int = 2,
                      dot_labels: str = "auto"
                      ) -> cartopy.mpl.geoaxes.GeoAxes:
        """
        This function plots seismicity on a surface. If ``include_map`` is
        set to ``True``, a nice natural earth map is used, otherwise the
        seismicity is just plotted on a blank grid. In the latter case,
        the grid is stretched according to the midpoint latitude.

        Args:
            resolution:     Resolution of the map, "10m", "50m" and "110m"
                        available.
            include_map:    If True, seismicity will be plotted on natural
                        earth map, otherwise it will be plotted on a blank
                        grid.
            country:        Name of country, if None map will fit to data
                        points.
            colors:         Color of background. If None is chosen, it will be
                        either white or standard natural earth colors.
            style:          Style of map, "satellite" or "street" are
                        available.
            dot_smallest:   Smallest dot size for magnitude scaling.
            dot_largest:    Largest dot size for magnitude scaling.
            dot_interpolation_power: Interpolation power for scaling.
            dot_labels:     Determines how labels for
                        magnitudes can be created. Input for matplotlib's
                        ``PathCollection.legend_elements``. If ``None``, no
                        label is shown. If an integer, target to use
                        ``dot_labels`` elements in the normed range.
                        If "auto", an automatic range is chosen for the
                        labels (default). If a list, uses elements of list
                        which are between minimum and maximum magnitude of
                        dataset for the legend.
                        Finally, a ``~.ticker.Locator`` can be provided to use
                        a predefined ``matplotlib.ticker`` (e.g.
                        ``FixedLocator``, which results in the same legend as
                        providing a list of values).
        Returns:
            ax: GeoAxis object
        """
        ax = plot_in_space(self.longitude,
                           self.latitude,
                           self.magnitude,
                           resolution=resolution,
                           include_map=include_map,
                           country=country,
                           colors=colors,
                           style=style,
                           dot_smallest=dot_smallest,
                           dot_largest=dot_largest,
                           dot_interpolation_power=dot_interpolation_power,
                           dot_labels=dot_labels)
        return ax

    @require_cols(require=['time', 'magnitude'])
    def plot_cum_count(self,
                       ax: plt.Axes | None = None,
                       mcs: np.ndarray = np.array([0]),
                       delta_m: float | None = None
                       ) -> plt.Axes:
        """
        Plots cumulative count of earthquakes in given catalog above given Mc
        through time. Plots a line for each given completeness magnitude in
        the array ``mcs``.

        Args:
            times:      Array containing times of events.
            magnitudes: Array of magnitudes of events corresponding to the
                    ``times``.
            ax:         Axis where figure should be plotted.
            mcs:        The list of completeness magnitudes for which we show
                    lines on the plot.
            delta_m:    Binning precision of the magnitudes.

        Returns:
            ax: Ax that was plotted on.
        """

        if delta_m is None:
            delta_m = self.delta_m
        ax = plot_cum_count(self.time,
                            self.magnitude,
                            ax=ax,
                            mcs=mcs,
                            delta_m=delta_m)
        return ax

    @require_cols(require=['time', 'magnitude'])
    def plot_mags_in_time(self,
                          ax: plt.Axes | None = None,
                          mc_change_times: list | None = None,
                          mcs: list | None = None,
                          dot_smallest: int = 10,
                          dot_largest: int = 200,
                          dot_interpolation_power: int = 2,
                          color_dots: str = "blue",
                          color_line: str = "#eb4034",
                          ) -> plt.Axes:
        """
        Creates a scatter plot, each dot is an event. Time shown on the x-axis,
        magnitude shown on the y-axis, but also reflected in the size of dots.

        Optionally, adds lines that represent the change in completeness
        magnitude. For example, ``mc_change_times = [2000, 2005]`` and
        ``mcs = [3.5, 3.0]`` means that between 2000 and 2005, Mc is 3.5
        and after 2005, Mc is 3.0.

        Args:
            times:      Array containing times of events.
            magnitudes: Array of magnitudes of events corresponding to the
                    ``times``.
            ax:         Axis where figure should be plotted.
            mc_change_times: List of points in time when Mc changes, sorted in
                    increasing order, can be given as a list of datetimes
                    or integers (years).
            mcs:        Changed values of Mc at times given in
                    ``mc_change_times``.
            dot_smallest: Smallest dot size for magnitude scaling.
            dot_largest: Largest dot size for magnitude scaling.
            dot_interpolation_power: Interpolation power for scaling.
            color_dots: Color of the dots representing the events.
            color_line: Color of the line representing the Mc changes.

        Returns:
            ax: ax that was plotted on
        """
        ax = plot_mags_in_time(self.time,
                               self.magnitude,
                               ax=ax,
                               mc_change_times=mc_change_times,
                               mcs=mcs,
                               dot_smallest=dot_smallest,
                               dot_largest=dot_largest,
                               dot_interpolation_power=dot_interpolation_power,
                               color_dots=color_dots,
                               color_line=color_line)
        return ax

    @require_cols(require=['magnitude'])
    def plot_cum_fmd(self,
                     ax: plt.Axes | None = None,
                     b_value: float | None = None,
                     mc: float | None = None,
                     delta_m: float = 0,
                     color: str | list = None,
                     size: int = None,
                     grid: bool = False,
                     bin_position: str = "center",
                     legend: bool | str | list = True
                     ) -> plt.Axes:
        """
        Plots cumulative frequency magnitude distribution, optionally with a
        corresponding theoretical Gutenberg-Richter (GR) distribution. The GR
        distribution is plotted provided the b-value is given.

        Args:
            magnitudes: Array of magnitudes.
            ax:         Axis where figure should be plotted.
            b_value:    The b-value of the theoretical GR distribution to plot.
            mc:         Completeness magnitude of the theoretical GR
                    distribution.
            delta_m:    Discretization of the magnitudes; important for the
                    correct visualization of the data.
            color:      Color of the data. If one value is given, it is used
                    for points, and the line of the theoretical GR distribution
                    if it is plotted. If a list of colors is given, the first
                    entry is the color of the points, and the second of the
                    line representing the GR distribution.
            size:       Size of the data points.
            grid:       Indicates whether or not to include grid lines.
            bin_position: Position of the bin, options are  'center' and 'left'
                    accordingly, left edges of bins or center points are
                    returned.

        Returns:
            ax: The ax object that was plotted on.
        """
        if delta_m is None:
            delta_m = self.delta_m
        if mc is None:
            mc = self.mc
        if b_value is None:
            b_value = self.b_value
        ax = plot_cum_fmd(self.magnitude,
                          ax=ax,
                          b_value=b_value,
                          mc=mc,
                          delta_m=delta_m,
                          color=color,
                          size=size,
                          grid=grid,
                          bin_position=bin_position,
                          legend=legend)
        return ax

    @require_cols(require=['magnitude'])
    def plot_fmd(self,
                 ax: plt.Axes | None = None,
                 delta_m: float = None,
                 color: str = None,
                 size: int = None,
                 grid: bool = False,
                 bin_position: str = "center",
                 legend: bool | str | list = True
                 ) -> plt.Axes:
        """
        Plots frequency magnitude distribution. If no binning is specified, the
        assumed value of ``delta_m`` is 0.1.

        Args:
            magnitudes:     Array of magnitudes.
            ax:             The axis where figure should be plotted.
            delta_m:        Discretization of the magnitudes, important for the
                        correct visualization of the data.
            color:          Color of the data.
            size:           Size of data points.
            grid:           Indicates whether or not to include grid lines.
            bin_position:   Position of the bin, options are  "center" and
                        "left" accordingly, left edges of bins or center points
                        are returned.

        Returns:
            ax: The ax object that was plotted on.
        """
        if delta_m is None:
            delta_m = self.delta_m
        ax = plot_fmd(self.magnitude,
                      ax=ax,
                      delta_m=delta_m,
                      color=color,
                      size=size,
                      grid=grid,
                      bin_position=bin_position,
                      legend=legend)

        return ax

    @require_cols(require=['magnitude'])
    def plot_mc_vs_b(self,
                     mcs: np.ndarray,
                     delta_m: float = None,
                     b_method: BValueEstimator = ClassicBValueEstimator,
                     confidence_interval: float = 0.95,
                     ax: plt.Axes | None = None,
                     color: str = "blue",
                     label: str | None = None,
                     *args,
                     **kwargs
                     ) -> plt.Axes:
        """
        Plots the estimated b-value in dependence of the completeness
        magnitude.

        Args:
            magnitudes:     Array of magnitudes.
            mcs:            Array of completeness magnitudes.
            delta_m:        Discretization of the magnitudes.
            b_method:       Method used for b-value estimation.
            confidence_interval: Confidence interval that should be plotted.
            ax:             Axis where figure should be plotted.
            color:          Color of the data.
            label:          Label of the data that will be put in the legend.
            **kwargs:       Additional keyword arguments for the b-value
                        estimator.

        Returns:
            ax: ax that was plotted on
        """

        if delta_m is None:
            delta_m = self.delta_m
        ax = plot_mc_vs_b(self.magnitude,
                          mcs,
                          delta_m,
                          b_method,
                          confidence_interval,
                          ax,
                          color,
                          label,
                          *args,
                          **kwargs)
        return ax

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

        Will fill in missing IDs with UUIDs, represented as strings.
        """
        df = self.copy()

        for col in ['eventID', 'originID', 'magnitudeID']:
            if col not in df.columns:
                df[col] = df.apply(lambda _: uuid.uuid4().hex, axis=1)

        mag_types = set([mag.split('_')[1]
                        for mag in df._secondary_magnitudekeys()])

        for mag_type in mag_types:
            if f'magnitude_{mag_type}_magnitudeID' not in df.columns:
                df[f'magnitude_{mag_type}_magnitudeID'] = \
                    df.apply(
                        lambda x: uuid.uuid4().hex
                        if not mag_type == x['magnitude_type']
                        else x['magnitudeID'], axis=1)

        return df

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID=' ', author=' ') -> str:
        """
        Convert the catalog to QuakeML format.

        Args:
            agencyID:   Agency ID with which to store the catalog.
            author:     Author of the catalog.

        Returns:
            catalog:    The catalog in QuakeML format.

        Examples:
            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import Catalog
                >>> simple_catalog = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'depth': [5.5, 10.52, 50.4],
                ...     'time': pd.to_datetime(['1900-01-01 05:05:13',
                ...                             '1982-04-07 07:07:15',
                ...                             '2020-11-30 12:30:59']),
                ...     'magnitude': [1.0, 2.5, 3.9],
                ...     'magnitude_type': ['Ml', 'Ml', 'Ml'],
                ...     })
                >>> simple_catalog.to_quakeml()
                <?xml version="1.0" encoding="UTF-8"?>
                <q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2"
                    xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
                ...
        """

        df = self.copy()
        df = df._create_ids()
        df = df.dropna(subset=['latitude', 'longitude', 'time'])
        if len(df) != len(self):
            self.logger.info(
                f"Dropped {len(self) - len(df)} rows with missing values")

        secondary_mags = self._secondary_magnitudekeys()

        data = dict(events=df.to_dict(orient='records'),
                    agencyID=agencyID, author=author)

        for event in data['events']:
            event['sec_mags'] = defaultdict(dict)
            for mag in secondary_mags:
                if pd.notna(event[mag]) and pd.notna(event['magnitude_type']) \
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
        data:           Data to initialize the catalog with.
        name:           Name of the catalog.
        n_catalogs:     Total number of catalogs represented,
                    including empty catalogs.
        args:           Additional arguments to pass to pandas
                    DataFrame constructor.
        starttime:      Start time of the catalog.
        endtime:        End time of the catalog.
        mc:             Completeness magnitude of the catalog.
        delta_m:        Magnitude binning of the catalog.
        kwargs:         Additional keyword arguments to pass to pandas
                    DataFrame constructor.

    See Also:
        The Catalog class is a subclass of :class:`pandas.DataFrame`, and
        inherits all of its methods and attributes.

        The ForecastCatalog class is a subclass of
        :class:`seismostats.Catalog`, and inherits all of its
        methods and attributes.
    """

    _required_cols = CATALOG_COLUMNS + ['catalog_id']
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
