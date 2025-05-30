from __future__ import annotations

import inspect
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from shapely import Polygon

from seismostats.analysis.avalue.base import AValueEstimator
from seismostats.analysis.avalue.classic import ClassicAValueEstimator
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.estimate_mc import (estimate_mc_b_stability,
                                              estimate_mc_ks, estimate_mc_maxc)
from seismostats.io.parser import parse_quakeml, parse_quakeml_file
from seismostats.plots.basics import (plot_cum_count, plot_cum_fmd, plot_fmd,
                                      plot_mags_in_time)
from seismostats.plots.seismicity import plot_in_space
from seismostats.plots.statistical import plot_mc_vs_b
from seismostats.utils import (_check_required_cols, _render_template,
                               _robust_parse_datetime, require_cols)
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
    '''
    A catalog of seismic events represented in tabular form, where
    each row corresponds to a single earthquake.

    To be a valid Catalog object, it must have at least a
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
            >>> from seismostats import Catalog
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
        :ivar a_value:      Gutenberg-Richter a-value of the catalog.
        :ivar starttime:    Start time of the catalog.
        :ivar endtime:      End time of the catalog.
        :ivar bounding_polygon: 2D boundary of the catalog.
        :ivar depth_min:    Minimum depth for which events are included in the
                        catalog.
        :ivar depth_max:    Maximum depth for which events are included in the
                        catalog.
        :ivar logger:       Logger for the catalog.
    '''

    _metadata = ['name', '_required_cols', 'mc', 'a_value',
                 'delta_m', 'b_value', 'starttime', 'endtime',
                 'bounding_polygon', 'depth_min', 'depth_max',
                 'logger']

    _required_cols = CATALOG_COLUMNS

    @property
    def _constructor(self):
        '''
        Required for subclassing Pandas DataFrame.
        '''
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
            a_value: float | None = None,
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
        self.a_value = a_value
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
                try:
                    self[tc] = pd.to_datetime(
                        self[tc], format='ISO8601').dt.tz_localize(None)
                except (ValueError, OutOfBoundsDatetime):
                    self[tc] = self[tc].apply(_robust_parse_datetime)

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
        '''
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
        '''
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
        '''
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
        '''

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
        '''
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

                >>> oq_cat = OQCatalog.make_from_dict({
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
                >>> cat = Catalog.from_openquake(oq_cat)
                >>> cat

                   longitude   latitude  depth                time  magnitude
                0      42.35    3.34444   5.50 1900-01-01 05:05:13        1.0
                1       1.35    5.13500  10.52 1982-04-07 07:07:15        2.5
                2       2.35    2.13400  50.40 2020-11-30 12:30:59        3.9
        '''
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
        '''
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
                >>> cat = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'depth': [5.5, 10.52, 50.4],
                ...     'time': pd.to_datetime(['1900-01-01 05:05:13',
                ...                             '1982-04-07 07:07:15',
                ...                             '2020-11-30 12:30:59']),
                ...     'magnitude': [1.0, 2.5, 3.9]
                ...     })
                >>> oq_cat = cat.to_openquake()
                >>> type(oq_catalog)

                <class 'openquake.hmtk.seismicity.catalogue.Catalogue'>
        '''
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
        '''
        Drop uncertainty columns from the catalog.

        Drops columns with names ending in '_uncertainty', '_lowerUncertainty',
        '_upperUncertainty', and '_confidenceLevel'.

        Returns:
            catalog: Catalog with uncertainty columns removed.
        '''

        rgx = "(_uncertainty|_lowerUncertainty|" \
            "_upperUncertainty|_confidenceLevel)$"

        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

    def drop_ids(self) -> Catalog:
        '''
        Drop event, origin, and magnitude IDs from the catalog.

        Drops columns named 'eventID', 'originID', and 'magnitudeID'.

        Returns:
            catalog: Catalog with ID columns removed.
        '''

        rgx = "(eventID|originID|magnitudeID)$"
        cols = self.filter(regex=rgx).columns
        df = self.drop(columns=cols)
        return df

    @require_cols(require=_required_cols)
    def strip(self, inplace: bool = False) -> Catalog | None:
        '''
        Remove all columns except the required ones
        defined in ``_required_cols``.

        Args:
            inplace:    Whether to perform the operation in place on the data.

        Returns:
            catalog:    Catalog with the stripped columns.
        '''
        df = self.drop(columns=set(self.columns).difference(
            set(self._required_cols)), inplace=inplace)
        if not inplace:
            return df

    @require_cols(require=['magnitude'])
    def bin_magnitudes(self, delta_m: float = None, inplace: bool = False) \
            -> Catalog | None:
        '''
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
                >>> cat = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.02, 2.53, 3.99]
                ...     })
                >>> cat.bin_magnitudes(delta_m=0.1)

                    longitude   latitude  magnitude
                0       42.35    3.34444        1.0
                1        1.35    5.13500        2.5
                2        2.35    2.13400        4.0
        '''
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
    def estimate_mc_maxc(
        self,
        fmd_bin: float,
        correction_factor: float | None = 0.2,
    ) -> tuple[float, dict[str, Any]]:
        '''
        Returns the completeness magnitude (mc) estimate using the maximum
        curvature method.

        Source:
            - Wiemer, S. and Wyss, M., 2000. Minimum magnitude of completeness
              in earthquake catalogs: Examples from Alaska, the western United
              States, and Japan. Bulletin of the Seismological Society of
              America, 90(4), pp.859-869.
            - Woessner, J. and Wiemer, S., 2005. Assessing the quality of
              earthquake catalogues: Estimating the magnitude of completeness
              and its uncertainty. Bulletin of the Seismological Society of
              America, 95(2), pp.684-698.

        Attention:
            ``Catalog.mc`` will be replaced by the ``mc`` return value.

        Args:
            magnitudes:         Array of magnitudes to test.
            fmd_bin:            Bin size for the maximum curvature method.
                        This can be independent ofthe descritization of the
                        magnitudes. The original value for the maximum
                        curvature method is 0.1. However, the user can decide
                        which value to use.
                        The optimal value would be as small as possible while
                        at the same time ensuring that there are enough
                        magnitudes in each bin. If the bin size is too small,
                        the method will not work properly.
            correction_factor:  Correction factor for the maximum curvature
                            method (default value after Woessner & Wiemer
                            2005).

        Returns:
            mc:                 Estimated completeness magnitude.
            mc_info:
                Dictionary with additional information about the calculation
                of the best ``mc``, including:

                - correction_factor:   Correction factor for the maximum
                  curvature method (default value +0.2 after Woessner
                  & Wiemer 2005).

        Examples:
            .. code-block:: python

                >>> from seismostats import Catalog
                >>> cat = Catalog.from_dict({
                ...     'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                ...                   1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                ...                   1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                ...                   1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                ...                   1.3, 1.1, 1.5, 1.4]})
                >>> cat.estimate_mc_maxc(fmd_bin=0.1)
                >>> cat.mc
                1.4

            The mc_maxc method also returns the correction factor used in the
            calculation of the best mc value.

            .. code-block:: python

                >>> best_mc, mc_info = cat.estimate_mc_maxc(fmd_bin=0.1)
                >>> mc_info['correction_factor']
                0.2

        '''
        self.mc, mc_info = estimate_mc_maxc(self.magnitude,
                                            fmd_bin=fmd_bin,
                                            correction_factor=correction_factor)
        return self.mc, mc_info

    @require_cols(require=['magnitude'])
    def estimate_mc_b_stability(
        self,
        delta_m: float | None = None,
        mcs_test: np.ndarray | None = None,
        stop_when_passed: bool = True,
        b_method: BValueEstimator = ClassicBValueEstimator,
        stability_range: float = 0.5,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[float | None, dict[str, Any]]:
        '''
        Estimates the completeness magnitude (mc) using b-value stability.

        The stability of the b-value is tested by default on half a magnitude
        unit (in line with the 5x0.1 in the orginial paper). Users can change
        the range for the stability test by changing the stability_range.

        Source:
            Woessner, J, and Stefan W. "Assessing the quality of earthquake
            catalogues: Estimating the magnitude of completeness and its
            uncertainty." Bulletin of the Seismological Society of America 95.2
            (2005): 684-698.

        Attention:
            ``Catalog.mc`` will be replaced by the ``best_mc`` return value.

        Args:
            delta_m:        Bin size of discretized magnitudes. Catalog
                        needs to be rounded to bins beforehand. Either given
                        as parameter or taken from the object attribute.
            mcs_test:       Array of tested completeness magnitudes. If None,
                        it will be generated automatically based on the
                        ``magnitudes`` and ``delta_m``
            stop_when_passed: Boolean that indicates whether to stop
                        computation when a completeness magnitude (mc) has
                        passed the test.
            b_method:       b-value estimator to use for b-value calculation.
            stability_range: Magnitude range to consider for the stability test.
                        Default compatible with the original definition of
                        Cao & Gao 2002.
            verbose:        Boolean that indicates whether to print verbose
                        output.
            **kwargs:       Additional parameters to be passed to the b-value
                        estimator.

        Returns:
            best_mc:        Best magnitude of completeness estimate.
            mc_info:
                Dictionary with additional information about the calculation
                of the best ``mc``, including:

                - best_b_value: b-value associated with ``best_mc``.
                - mcs_tested:     Array of tested completeness magnitudes.
                - b_values_tested: Array of b-values associated to tested mcs.
                - diff_bs:      Array of differences divided by std, associated
                  with tested mcs. If a value is smaller than one,
                  this means that the stability criterion is met.

        Examples:
            .. code-block:: python

                >>> from seismostats import Catalog
                >>> cat = Catalog.from_dict({
                ...     'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                ...                   1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                ...                   1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                ...                   1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                ...                   1.3, 1.1, 1.5, 1.4]})
                >>> cat.delta_m = 0.1
                >>> cat.estimate_mc_b_stability()
                >>> cat.mc

                1.1

            The mc_b_stability method returns additional information about the
            calculation of the best mc, like b-values tested and the array of
            differences. Those are returned by the method and can be used for
            further analysis.

            .. code-block:: python

                >>> best_mc, mc_info = cat.estimate_mc_b_stability()
                >>> (mc_info['mcs_tested'], mc_info['diff_bs'])

                (array([1. , 1.1]), [2.23375277112158, 0.9457747650207577])
        '''
        if delta_m is None and self.delta_m is None:
            raise ValueError("Binning (delta_m) needs to be set.")
        if delta_m is None:
            delta_m = self.delta_m

        self.mc, mc_info = \
            estimate_mc_b_stability(self.magnitude,
                                    delta_m=delta_m,
                                    mcs_test=mcs_test,
                                    stop_when_passed=stop_when_passed,
                                    b_method=b_method,
                                    stability_range=stability_range,
                                    verbose=verbose,
                                    **kwargs)

        return (self.mc, mc_info)

    @require_cols(require=['magnitude'])
    def estimate_mc_ks(
        self,
        delta_m: float | None = None,
        mcs_test: list | None = None,
        p_value_pass: float = 0.1,
        stop_when_passed: bool = True,
        b_value: float | None = None,
        b_method: BValueEstimator = ClassicBValueEstimator,
        n: int = 10000,
        ks_ds_list: list[list] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[float | None, dict[str, Any]]:
        '''
        Returns the smallest magnitude in a given list of completeness
        magnitudes for which the KS test is passed, i.e., where the null
        hypothesis that the sample of magnitudes is drawn from a
        Gutenberg-Richter law cannot be rejected.

        Source:
            - Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
              distributions in empirical data. SIAM review, 51(4), pp.661-703.
            - Mizrahi, L., Nandan, S. and Wiemer, S., 2021. The effect of
              declustering on the size distribution of mainshocks. Seismological
              Society of America, 92(4), pp.2333-2342.

        Attention:
            ``Catalog.mc`` will be replaced by the ``best_mc`` return value.

        Args:
            delta_m:        Bin size of discretized magnitudes. Catalog
                        needs to be rounded to bins beforehand. Either given
                        as parameter or taken from the object attribute.
            mcs_test:       Array of tested completeness magnitudes. If `None`,
                        it will be generated automatically based on `magnitudes`
                        and `delta_m`.
            p_value_pass:   p-value required to pass the test.
            stop_when_passed:  Stop calculations when first mc passes the test.
            b_value:        If `b_value` is 'known', only estimate `mc` assuming
                        the given `b_value`. If `None`, the b-value is either
                        taken from the object attribute or estimated.
            b_method:       b-value estimator to use if b-value needs to be
                        calculated from data.
            n:              Number of number of times the KS distance is
                        calculated for estimating the p-value.
            ks_ds_list:     KS distances from synthetic data with the given
                        parameters. If `None`, they will be estimated here.
            verbose:           Boolean that indicates whether to print verbose
                        output.
            **kwargs:       Additional parameters to be passed to the b-value
                        estimator.
        Returns:
            best_mc:        ``mc`` for which the p-value is lowest.
            mc_info:
                Dictionary with additional information about the calculation
                of the best ``mc``, including:

                - best_b_value: ``b_value`` corresponding to the best ``mc``.
                - mcs_tested: Tested completeness magnitudes.
                - b_values_tested: Tested b-values.
                - ks_ds: KS distances.
                - p_values: Corresponding p-values.

        Examples:
            .. code-block:: python

                >>> from seismostats import Catalog
                >>> cat = Catalog.from_dict({
                ...     'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                ...                   1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                ...                   1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                ...                   1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                ...                   1.3, 1.1, 1.5, 1.4]})
                >>> cat.delta_m = 0.1
                >>> cat.estimate_mc_ks()
                >>> cat.mc

                1.0

            The mc_ks method returns additional information about the
            calculation of the best mc, like b-values tested and ks
            distances. Those are returned by the method and can be
            used for further analysis.

            .. code-block:: python

                >>> best_mc, mc_info = cat.estimate_mc_ks()
                >>> (mc_info['b_values_tested'], mc_info['ks_ds'])

                ([0.9571853220063774], [0.1700244200244202])
        '''
        if delta_m is None and self.delta_m is None:
            raise ValueError("Binning (delta_m) needs to be set.")
        if delta_m is None:
            delta_m = self.delta_m

        if b_value is None and self.b_value is not None:
            b_value = self.b_value

        best_mc, mc_info = \
            estimate_mc_ks(self.magnitude,
                           delta_m=delta_m,
                           mcs_test=mcs_test,
                           p_value_pass=p_value_pass,
                           stop_when_passed=stop_when_passed,
                           b_value=b_value,
                           b_method=b_method,
                           n=n,
                           ks_ds_list=ks_ds_list,
                           verbose=verbose,
                           **kwargs)

        self.mc = best_mc

        return (best_mc, mc_info)

    @require_cols(require=['magnitude'])
    def estimate_b(
        self,
        mc: float | None = None,
        delta_m: float | None = None,
        weights: list | None = None,
        method: BValueEstimator = ClassicBValueEstimator,
        **kwargs
    ) -> BValueEstimator:
        '''
        Estimates b-value of the Gutenberg-Richter (GR) law, using the
        magnitudes in the `Catalog`. Sets attribute b-value to the computed
        value, but also returns the b-value estimator object.

        Args:
            mc:         Completeness magnitude, either given as parameter or
                    taken from the object attribute.
            delta_m:    Discretization of magnitudes, either given as parameter
                    or taken from the object attribute.
            weights:    Weights of each magnitude can be specified here.
            method:     BValueEstimator class to use for calculation.
            **kwargs:   Additional parameters to be passed to the
                    `BValueEstimator.calculate` method.

        Returns:
            estimator: Object of type
                    :func:`~seismostats.analysis.bvalue.classic.ClassicBValueEstimator`
                    or of the type provided by the `method` parameter.

        See Also:
            By default uses
            :func:`~seismostats.analysis.bvalue.classic.ClassicBValueEstimator`
            as estimator. All available estimators can be found in the
            :ref:`bvalues <bvalues>` module.

        Examples:
            The `estimate_b` method sets the `b_value` attribute of
            the catalog to the computed value.

            .. code-block:: python

                >>> from seismostats import Catalog
                >>> cat = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'magnitude': [1.0, 2.5, 3.9]
                ...     })
                >>> cat.estimate_b(mc=1.0, delta_m=0.1)
                >>> cat.b_value

                0.28645181449530005

            The returned estimator can be used to access the remaining results,
            see the documentation of
            :func:`~seismostats.analysis.bvalue.classic.ClassicBValueEstimator`
            or the explicitly used :ref:`estimator <bvalues>` for more
            information.

            .. code-block:: python

                >>> estimator = cat.estimate_b(mc=1.0, delta_m=0.1)
                >>> estimator.beta, estimator.std

                (0.6595796779179737, 0.15820210898689366)

            Using for example the
            :func:`~seismostats.analysis.bvalue.positive.BPositiveBValueEstimator`,
            the ``time`` parameter can either be passed in the ``kwargs``,
            like the additional ``dmc`` parameter. If not passed, it will be
            taken from the catalog columns directly.

            .. code-block:: python

                >>> from datetime import datetime
                >>> from seismostats.analysis import BPositiveBValueEstimator
                >>> cat['time'] = [datetime(2000, 1, 1),
                ...                           datetime(2000, 1, 2),
                ...                           datetime(2000, 1, 3)]
                >>> estimator = cat.estimate_b(mc=1.0, delta_m=0.1,
                ...     method=BPositiveBValueEstimator, dmc=0.3)
                >>> type(estimator)

                <class 'seismostats.analysis.BPositiveBValueEstimator'>
        '''
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

        # check catalog columns for additional argument values.
        col_kwargs = _check_catalog_cols(
            method.calculate, BValueEstimator.calculate, self)
        col_kwargs.update(**kwargs)

        # Create and call the estimator
        b_estimator = method()
        b_estimator.calculate(self.magnitude,
                              mc,
                              delta_m,
                              weights=weights,
                              **col_kwargs)

        self.b_value = b_estimator.b_value

        return b_estimator

    @require_cols(require=['magnitude'])
    def estimate_a(self,
                   mc: float | None = None,
                   delta_m: float | None = None,
                   scaling_factor: float | None = None,
                   m_ref: float | None = None,
                   b_value: float | None = None,
                   method: AValueEstimator = ClassicAValueEstimator,
                   **kwargs) -> AValueEstimator:
        '''

        Estimates a-value of the Gutenberg-Richter (GR) law, using the
        magnitudes in the `Catalog`. Sets attribute a-value to the computed
        value, but also returns the a-value estimator object.

        Args:
            magnitudes:     Array of magnitudes.
            mc:             Completeness magnitude.
            delta_m:        Bin size of discretized magnitudes.
            scaling_factor: Scaling factor.
                        If given, this is used to normalize the number of
                        observed events. For example: Volume or area of the
                        region considered or length of the time interval,
                        given in the unit of interest.
            m_ref:          Reference magnitude for which the a-value
                        is estimated.
            b_value:        b-value of the Gutenberg-Richter law. Only relevant
                        when `m_ref` is not `None`.
            method:         AValueEstimator class to use for calculation.
            **kwargs:       Additional parameters to be passed to the
                        `AValueEstimator.calculate` method.

        Returns:
            estimator: Object of type
                    :func:`~seismostats.analysis.avalue.classic.ClassicAValueEstimator`
                    or of the type provided by the `method` parameter.

        See Also:
            By default uses
            :func:`~seismostats.analysis.avalue.classic.ClassicAValueEstimator`
            as estimator. All available estimators can be found in the
            :ref:`avalues <avalues>` module.

        Examples:
            The `estimate_a` method sets the `a_value` attribute of
            the catalog to the computed value.

            .. code-block:: python

                >>> from datetime import datetime
                >>> from seismostats import Catalog

                >>> cat = Catalog.from_dict({
                ...         'magnitude': [0, 0.9, -1, 0.2, 0.5],
                ...         'time': [datetime(2000, 1, 1),
                ...                  datetime(2000, 1, 2),
                ...                  datetime(2000, 1, 3),
                ...                  datetime(2000, 1, 4),
                ...                  datetime(2000, 1, 5)]})
                >>> cat.mc = -1.0

                >>> cat.estimate_a(delta_m=0.1)
                >>> cat.a_value

                0.6989700043360189

            The returned estimator can be used to access the remaining results,
            see the documentation of
            :func:`~seismostats.analysis.avalue.classic.ClassicAValueEstimator`
            or the explicitly used :ref:`estimator <avalues>` for more
            information.

            .. code-block:: python

                >>> estimator = cat.estimate_a(delta_m=0.1)
                >>> estimator.a_value, estimator.mc

                (0.6989700043360189, -1.0)

            Using for example the
            :func:`~seismostats.analysis.avalue.positive.APositiveAValueEstimator`,
            the ``time`` parameter can either be passed in the ``kwargs``, like
            the additional ``dmc`` parameter. If not passed, it will be taken
            from the catalog columns directly.

            .. code-block:: python

                >>> from seismostats.analysis import APositiveAValueEstimator
                >>> estimator = cat.estimate_a(delta_m=0.1,
                ...                   method=APositiveAValueEstimator, dmc=0.1)
                >>> type(estimator)

                <class 'seismostats.analysis.APositiveAValueEstimator'>
        '''

        if mc is None and self.mc is None:
            raise ValueError("Completeness magnitude (mc) needs to be set.")
        if mc is None:
            mc = self.mc

        if delta_m is None and self.delta_m is None:
            raise ValueError("Binning (delta_m) needs to be set.")
        if delta_m is None:
            delta_m = self.delta_m

        # check catalog columns for additional argument values.
        col_kwargs = _check_catalog_cols(
            method.calculate, AValueEstimator.calculate, self)
        col_kwargs.update(**kwargs)

        a_estimator = method()
        a_estimator.calculate(self.magnitude,
                              mc,
                              delta_m,
                              scaling_factor=scaling_factor,
                              m_ref=m_ref,
                              b_value=b_value,
                              **col_kwargs
                              )

        self.a_value = a_estimator.a_value

        return a_estimator

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
        '''
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
        '''
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
                       mcs: np.ndarray = np.array([0]),
                       delta_m: float | None = None,
                       ax: plt.Axes | None = None,
                       ) -> plt.Axes:
        '''
        Plots cumulative count of earthquakes in given catalog above given Mc
        through time. Plots a line for each given completeness magnitude in
        the array ``mcs``.

        Args:
            times:      Array containing times of events.
            magnitudes: Array of magnitudes of events corresponding to the
                    ``times``.
            mcs:        The list of completeness magnitudes for which we show
                    lines on the plot.
            delta_m:    Binning precision of the magnitudes.
            ax:         Axis where figure should be plotted.

        Returns:
            ax: Ax that was plotted on.
        '''

        if delta_m is None:
            delta_m = self.delta_m

        ax = plot_cum_count(self.time,
                            self.magnitude,
                            mcs=mcs,
                            delta_m=delta_m,
                            ax=ax)
        return ax

    @require_cols(require=['time', 'magnitude'])
    def plot_mags_in_time(self,
                          mc_change_times: list | None = None,
                          mcs: list | None = None,
                          ax: plt.Axes | None = None,
                          dot_smallest: int = 10,
                          dot_largest: int = 200,
                          dot_interpolation_power: int = 2,
                          color_dots: str = "blue",
                          color_line: str = "#eb4034",
                          ) -> plt.Axes:
        '''
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
            mc_change_times: List of points in time when Mc changes, sorted in
                    increasing order, can be given as a list of datetimes
                    or integers (years).
            mcs:        Changed values of Mc at times given in
                    ``mc_change_times``.
            ax:         Axis where figure should be plotted.
            dot_smallest: Smallest dot size for magnitude scaling.
            dot_largest: Largest dot size for magnitude scaling.
            dot_interpolation_power: Interpolation power for scaling.
            color_dots: Color of the dots representing the events.
            color_line: Color of the line representing the Mc changes.

        Returns:
            ax: ax that was plotted on
        '''
        ax = plot_mags_in_time(self.time,
                               self.magnitude,
                               mc_change_times=mc_change_times,
                               mcs=mcs,
                               ax=ax,
                               dot_smallest=dot_smallest,
                               dot_largest=dot_largest,
                               dot_interpolation_power=dot_interpolation_power,
                               color_dots=color_dots,
                               color_line=color_line)
        return ax

    @require_cols(require=['magnitude'])
    def plot_cum_fmd(self,
                     mc: float | None = None,
                     delta_m: float = None,
                     b_value: float | None = None,
                     ax: plt.Axes | None = None,
                     color: str | list = None,
                     size: int = None,
                     grid: bool = False,
                     bin_position: str = "center",
                     legend: bool | str | list = True
                     ) -> plt.Axes:
        '''
        Plots cumulative frequency magnitude distribution, optionally with a
        corresponding theoretical Gutenberg-Richter (GR) distribution. The GR
        distribution is plotted provided the b-value is given.

        Args:
            magnitudes: Array of magnitudes.
            mc:         Completeness magnitude of the theoretical GR
                    distribution.
            delta_m:    Discretization of the magnitudes; important for the
                    correct visualization of the data.
            b_value:    The b-value of the theoretical GR distribution to plot.
            ax:         Axis where figure should be plotted.
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
        '''
        if delta_m is None:
            delta_m = self.delta_m
        if mc is None:
            mc = self.mc
        if b_value is None:
            b_value = self.b_value
        ax = plot_cum_fmd(self.magnitude,
                          b_value=b_value,
                          mc=mc,
                          delta_m=delta_m,
                          ax=ax,
                          color=color,
                          size=size,
                          grid=grid,
                          bin_position=bin_position,
                          legend=legend)
        return ax

    @require_cols(require=['magnitude'])
    def plot_fmd(self,
                 fmd_bin: float,
                 ax: plt.Axes | None = None,
                 color: str = None,
                 size: int = None,
                 grid: bool = False,
                 bin_position: str = "center",
                 legend: bool | str | list = True
                 ) -> plt.Axes:
        '''
        Plots frequency magnitude distribution.

        Args:
            magnitudes:     Array of magnitudes.
            fmd_bin:        Bin size of magnitudes for plotting purposes.
            ax:             The axis where figure should be plotted.
            color:          Color of the data.
            size:           Size of data points.
            grid:           Indicates whether or not to include grid lines.
            bin_position:   Position of the bin, options are  "center" and
                        "left" accordingly, left edges of bins or center points
                        are returned.

        Returns:
            ax: The ax object that was plotted on.
        '''
        ax = plot_fmd(self.magnitude,
                      fmd_bin=fmd_bin,
                      ax=ax,
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
                     **kwargs
                     ) -> plt.Axes:
        '''
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
            **kwargs:       Additional parameters to be passed to the b-value
                        estimator.

        Returns:
            ax: ax that was plotted on
        '''

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
                          **kwargs)
        return ax

    def _secondary_magnitudekeys(self) -> list[str]:
        '''
        Get a list of secondary magnitude keys in the catalog.

        This will always include also the preferred magnitude type.

        Returns:
            keys: List of secondary magnitude keys in the catalog.
        '''

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
        '''
        Create missing event, origin, and magnitude IDs for the catalog.

        Will fill in missing IDs with UUIDs, represented as strings.
        '''
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
        '''
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
                >>> cat = Catalog.from_dict({
                ...     'longitude': [42.35, 1.35, 2.35],
                ...     'latitude': [3.34444, 5.135, 2.134],
                ...     'depth': [5.5, 10.52, 50.4],
                ...     'time': pd.to_datetime(['1900-01-01 05:05:13',
                ...                             '1982-04-07 07:07:15',
                ...                             '2020-11-30 12:30:59']),
                ...     'magnitude': [1.0, 2.5, 3.9],
                ...     'magnitude_type': ['Ml', 'Ml', 'Ml'],
                ...     })
                >>> cat.to_quakeml()
                <?xml version="1.0" encoding="UTF-8"?>
                <q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2"
                    xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">
                ...
        '''

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
        ''' Propagate metadata from other to self.
            Source: https://github.com/geopandas/geopandas

        Args:
            other:  The other object to finalize with.
            method: The method used to finalize the objects.
            kwargs: Additional keyword arguments.

        Returns:
            self:   The finalized object.
        '''
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
    '''
    A catalog of seismic events represented in tabular form, where
    each row corresponds to a single earthquake.
    The ForecastCatalog extends this structure to represent multiple
    realizations of the same catalog, distinguished by an additional
    column `catalog_id`.

    To be a valid ForecastCatalog object, it must have the
    following columns: `longitude`, `latitude`, `depth`, `time`,
    `magnitude`, `catalog_id`.

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
    '''

    _required_cols = CATALOG_COLUMNS + ['catalog_id']
    _metadata = Catalog._metadata + ['n_catalogs']

    def __init__(self, data=None, *args, n_catalogs=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        # Total number of catalogs represented, including empty catalogs
        self.n_catalogs = n_catalogs

    @require_cols(require=_required_cols)
    def to_quakeml(self, agencyID: str = ' ', author: str = ' ') -> list[str]:
        '''
        Convert the catalogs to QuakeML format.

        Args:
            agencyID:   Agency ID.
            author:     Author of the catalog.

        Returns:
            catalogs:   List of catalogs in QuakeML format.
        '''
        catalogs = []
        for _, group in self.groupby('catalog_id'):
            catalogs.append(Catalog(group).to_quakeml(agencyID, author))
        return catalogs


def _check_catalog_cols(method: Callable,
                        reference: Callable,
                        catalog: Catalog) -> dict:
    '''
    Check if any of the parameters of the `method`, which are not
    part of the parameters of the `reference` method, are present
    as columns in the DataFrame. If so, collect them and return them as
    keyword arguments.
    Also checks for singular form of the extra arguments.
    '''
    sig = inspect.signature(method).parameters.keys()
    base_args = inspect.signature(reference).parameters.keys()

    extra_args = [name for name in sig if name not in base_args]

    # singular form version of the extra arguments
    sing_args = [item if not item.endswith('s') else
                 item[:-1] for item in extra_args]

    # create list to check against
    check_list = zip(extra_args + extra_args, sing_args + extra_args)

    # Collect the matching arguments from the DataFrame
    col_kwargs = {arg: catalog[sarg] for arg, sarg in check_list
                  if sarg in catalog.columns}

    return col_kwargs
