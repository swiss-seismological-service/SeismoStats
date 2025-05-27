from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from seismostats.utils import _check_required_cols, require_cols

REQUIRED_COLS_RATEGRID = [
    'longitude_min', 'longitude_max',
    'latitude_min', 'latitude_max',
    'depth_min', 'depth_max',
    'a', 'b', 'mc'
]


def _rategrid_constructor_with_fallback(*args, **kwargs):
    df = GRRateGrid(*args, **kwargs)
    if not _check_required_cols(df, REQUIRED_COLS_RATEGRID):
        return pd.DataFrame(*args, **kwargs)
    if not _check_required_cols(df, required_cols=['grid_id']):
        return df
    return ForecastGRRateGrid(*args, **kwargs)


class GRRateGrid(pd.DataFrame):
    """
    Represents seismicity as a grid of cells, each with its own
    Gutenberg-Richter parameters (a-value, b-value, and mc).

    To be a valid RateGrid object, the it must have the following
    columns: longitude_min, longitude_max, latitude_min, latitude_max,
    depth_min, depth_max, a, b, and mc.

    Args:
        data:       Data to initialize the catalog with.
        name:       Name of the catalog.
        starttime:  Start time of the catalog. If a string, it must be in a
                    format that can be parsed by pandas.to_datetime.
        endtime:    End time of the catalog. If a string, it must be in a
                    format that can be parsed by pandas.to_datetime.
        kwargs:     Additional arguments and keyword arguments to pass to pandas
                    DataFrame constructor.

    Notes:
        The RateGrid class is a subclass of :class:`pandas.DataFrame`, and
        inherits all of its methods and attributes.

    Examples:
        Create a `GRRateGrid` from a dictionary.

        .. code-block:: python

            >>> import pandas as pd
            >>> from seismostats import GRRateGrid
            >>> data = {'longitude_min': [9.0, 10.0, 11.0],
            ...         'longitude_max': [10.0, 11.0, 12.0],
            ...         'latitude_min': [45.0, 46.0, 47.0],
            ...         'latitude_max': [46.0, 47.0, 48.0],
            ...         'depth_min': [10, 20, 30], 'depth_max': [20, 30, 40],
            ...         'a': [0, 1, 2], 'b': [0, 1, 2], 'mc': [0, 1, 2]}
            >>> rategrid = GRRateGrid(data,
            ...                       starttime=pd.Timestamp("2023-01-01"),
            ...                       endtime=pd.Timestamp("2023-01-02"))
            >>> rategrid

              longitude_min  longitude_max   ...  a   b   mc
            1           9.0           10.0   ...  0   0   0
            2          10.0           11.0   ...  1   1   1
            3          11.0           12.0   ...  2   2   2
            [3 rows x 9 columns]


        :ivar name:         Name of the GRRateGrid.
        :ivar starttime:    Start time of the GRRateGrid.
        :ivar endtime:      End time of the GRRateGrid.
    """
    _metadata = ['name', '_required_cols', 'starttime', 'endtime']
    _required_cols = REQUIRED_COLS_RATEGRID

    def __init__(self, data=None, *args, name=None,
                 starttime: pd.Timestamp | None = None,
                 endtime: pd.Timestamp | None = None,
                 **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

        self.starttime = starttime if isinstance(
            starttime, pd.Timestamp) else pd.to_datetime(starttime)

        self.endtime = endtime if isinstance(
            endtime, pd.Timestamp) else pd.to_datetime(endtime)

        if len(self.index.names) > 1:
            try:
                self._reindex_cell_id()
            except AttributeError:
                pass

    @property
    def _constructor(self):
        return _rategrid_constructor_with_fallback

    @classmethod
    def concat(cls, grids: list[GRRateGrid]):
        """
        Concatenate a list of GRRateGrid objects into a single GRRateGrid
        with a MultiIndex using starttime, optionally endtime and a unique
        cell number.

        Requires that all GRRateGrid objects have the same columns and
        that at least the starttime attribute is set for all of them.
        Args:
            grids:  List of GRRateGrid objects to concatenate.

        Returns:
            rategrid: A new GRRateGrid object containing the concatenated data.

        Examples:
            Create two `GRRateGrid` objects with different time ranges
            and then concatenate them into a single `GRRateGrid` with
            a MultiIndex based on starttime, endtime and cell_id.

            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import GRRateGrid
                >>> data = {'longitude_min': [9.0, 10.0, 11.0],
                ...         'longitude_max': [10.0, 11.0, 12.0],
                ...         'latitude_min': [45.0, 46.0, 47.0],
                ...         'latitude_max': [46.0, 47.0, 48.0],
                ...         'depth_min': [1, 2, 3], 'depth_max': [2, 3, 4],
                ...         'a': [0, 1, 2], 'b': [0, 1, 2], 'mc': [0, 1, 2]}
                >>> rategrid1 = GRRateGrid(
                ...     data,
                ...     starttime=pd.Timestamp("2023-01-01"),
                ...     endtime=pd.Timestamp("2023-01-02"))
                >>> rategrid2 = GRRateGrid(
                ...     data,
                ...     starttime=pd.Timestamp("2023-01-02"),
                ...     endtime=pd.Timestamp("2023-01-03"))
                >>> rategrid = GRRateGrid.concat([rategrid1, rategrid2])
                >>> rategrid

                                            longitude_min  ...  a  b  mc
                starttime  endtime    cell_id
                2023-01-01 2023-01-02 0               9.0  ...  0  0   0
                                      1              10.0  ...  1  1   1
                                      2              11.0  ...  2  2   2
                2023-01-02 2023-01-03 0               9.0  ...  0  0   0
                                      1              10.0  ...  1  1   1
                                      2              11.0  ...  2  2   2

        """
        if all((g.index.nlevels == 1 for g in grids)):
            endtime = all((g.endtime for g in grids))
            return cls(pd.concat([g.add_time_index(endtime) for g in grids]))
        elif all((g.index.nlevels == 2 for g in grids)) or \
                all((g.index.nlevels == 3 for g in grids)):
            return cls(pd.concat(grids, axis=0))
        else:
            raise ValueError("All grids must have the same number "
                             "of index levels (1, 2 or 3).")

    @require_cols(require=_required_cols)
    def strip(self, inplace: bool = False) -> GRRateGrid | None:
        """
        Remove all columns except the required ones for a GRRateGrid.

        Args:
            inplace:    Whether to perform the operation in place on the data.

        Returns:
            rategrid:    GRRateGrid with the stripped columns.

        Examples:
            Create a `GRRateGrid` with additional columns and then
            strip it to only keep the required columns.

            .. code-block:: python

                >>> from seismostats import GRRateGrid
                >>> rategrid = GRRateGrid(
                >>>     columns=['longitude_min', 'longitude_max',
                >>>              'latitude_min', 'latitude_max',
                >>>              'depth_min', 'depth_max', 'a',
                >>>              'b', 'mc', 'col1', 'col2'])
                >>> rategrid = rategrid.strip()
                >>> rategrid.columns

                Index(['longitude_min', 'longitude_max', 'latitude_min',
                       'latitude_max', 'depth_min', 'depth_max', 'a',
                       'b', 'mc'],
                      dtype='object')
        """
        df = self.drop(columns=set(self.columns).difference(
            set(self._required_cols)), inplace=inplace)

        if not inplace:
            return df

    @require_cols(require=_required_cols)
    def add_time_index(self, endtime=True) -> GRRateGrid:
        """
        Create MultiIndex using starttime and endtime, both taken from
        the object attributes, and a cell number for each spatial block.

        Args:
            endtime:    If True, create MultiIndex with starttime and endtime.
                        Otherwise, create MultiIndex with only starttime.

        Returns:
            rategrid:   A new RateGrid with the MultiIndex set.

        Examples:
            Create a `GRRateGrid`, then add a time index to it.

            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import GRRateGrid
                >>> data = {'longitude_min': [9.0, 10.0, 11.0],
                ...         'longitude_max': [10.0, 11.0, 12.0],
                ...         'latitude_min': [45.0, 46.0, 47.0],
                ...         'latitude_max': [46.0, 47.0, 48.0],
                ...         'depth_min': [1, 2, 3], 'depth_max': [2, 3, 4],
                ...         'a': [0, 1, 2], 'b': [0, 1, 2], 'mc': [0, 1, 2]}
                >>> rategrid = GRRateGrid(
                ...     data,
                ...     starttime=pd.Timestamp("2023-01-01"),
                ...     endtime=pd.Timestamp("2023-01-02"))
                >>> rategrid = rategrid.add_time_index()
                >>> rategrid

                                               longitude_min  ...  a  b  mc
                starttime  endtime    cell_id
                2023-01-01 2023-01-02 0                  9.0  ...  0  0   0
                                      1                 10.0  ...  1  1   1
                                      2                 11.0  ...  2  2   2

        """
        if not getattr(self, 'starttime', None):
            raise AttributeError(
                'Starttime must be set to use this method.')
        if not getattr(self, 'endtime', None) and endtime:
            raise AttributeError(
                'Endtime must be set to use this method.')

        df = self.copy()

        index = (df.starttime, df.endtime) if endtime else df.starttime
        names = ['starttime', 'endtime'] if endtime else ['starttime']

        # rename the index to cell_id, will be set in constructor
        df.index.name = 'cell_id'

        df = pd.concat({index: df}, names=names)

        return df

    @require_cols(require=_required_cols)
    def _reindex_cell_id(self) -> None:
        """
        If the RateGrid has a MultiIndex which includes `cell_id`
        as a level, this method will update the RateGrid's index to use
        unique cell_id values.
        """

        if 'cell_id' in self.index.names:
            cell_bounds = self[['longitude_min', 'longitude_max',
                                'latitude_min', 'latitude_max',
                                'depth_min', 'depth_max']]

            self['cell'] = np.unique(
                cell_bounds, axis=0, return_inverse=True, equal_nan=True)[1]

            self.set_index('cell', append=True, drop=True, inplace=True)
            self.index = self.index.droplevel('cell_id')

            self.index.set_names('cell_id', level='cell', inplace=True)

        if 'starttime' in self.index.names:
            self.starttime = self.index.get_level_values('starttime').min()
            if 'endtime' in self.index.names:
                self.endtime = self.index.get_level_values('endtime').max()
            else:
                self.endtime = self.index.get_level_values('starttime').max()

    def __finalize__(self, other, method=None, **kwargs):
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

        # concat operation: using metadata of the first object
        if method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(
                    other.objs[0], name, None))
        return self


class ForecastGRRateGrid(GRRateGrid):
    """
    A seismicity forecast on a grid where for each grid cell, the GR parameters
    (a-value, b-value, mc) are defined. Additionally to the GRRateGrid, this
    class has a grid_id column to identify each possible realization of the
    grid.

    To be a valid RateGrid object, it must have the following
    columns: `longitude_min`, `longitude_max`, `latitude_min`, `latitude_max`,
    `depth_min`, `depth_max`, `number_events`, `a`, `b`, `mc`, and `grid_id`.

    Args:
        data:       Data to initialize the catalog with.
        name:       Name of the catalog.
        starttime:  Start time of the catalog. If a string, it must be in a
                    format that can be parsed by `pandas.to_datetime`.
        endtime:    End time of the catalog. If a string, it must be in a
                    format that can be parsed by `pandas.to_datetime`.
        kwargs: Additional arguments and keyword arguments to pass to pandas
                DataFrame constructor.

    Notes:
        The ForecastRateGrid class is a subclass of :class:`pandas.DataFrame`,
        and inherits all of its methods and attributes.

    Examples:
        Create a ForecastGRRateGrid from a dictionary with two grid cells,
        each having two realizations (as indicated by `grid_id`).

        .. code-block:: python

            >>> import pandas as pd
            >>> from seismostats import ForecastGRRateGrid
            >>> data = {
            ...     'longitude_min': [9, 9, 10, 10],
            ...     'longitude_max': [10, 10, 11, 11],
            ...     'latitude_min': [45, 45, 46, 46],
            ...     'latitude_max': [46, 46, 47, 47],
            ...     'depth_min': [10, 10, 20, 20],
            ...     'depth_max': [20, 20, 30, 30],
            ...     'number_events': [5, 6, 10, 12],
            ...     'a': [0.8, 0.9, 1.0, 1.1],
            ...     'b': [0.95, 1.0, 1.05, 1.1],
            ...     'mc': [1.2, 1.2, 1.3, 1.3],
            ...     'grid_id': [0, 0, 1, 1]
            ... }
            >>> forecast = ForecastGRRateGrid(
            ...     data,
            ...     starttime=pd.Timestamp('2023-01-01'),
            ...     endtime=pd.Timestamp('2023-01-02'))
            >>> forecast

               longitude_min longitude_max  ...    a     b   mc  grid_id
            0            9.0          10.0  ...  0.8  0.95  1.2        0
            1            9.0          10.0  ...  0.9  1.00  1.2        0
            2           10.0          11.0  ...  1.0  1.05  1.3        1
            3           10.0          11.0  ...  1.1  1.10  1.3        1

    """

    _required_cols = REQUIRED_COLS_RATEGRID + ['grid_id']
    _metadata = GRRateGrid._metadata + ['n_grids']

    def __init__(self, data=None, *args, n_grids=None, **kwargs):
        super().__init__(data, *args, **kwargs)
        # Total number of catalogs represented, including empty catalogs
        self.n_grids = n_grids

    def calculate_statistics(self,
                             agg_func: str | Callable = 'mean',
                             extra_stats: dict = {}):
        """
        Get statistics for `a`, `b`, `alpha`, `mc` and `number_events`,
        if present, per timestep and grid cell, aggregated over all
        realizations of the grid (i.e. over all grid_id values).

        Works with a normal range index (aggregates over grid cells) or
        with a MultiIndex with `starttime`, (`endtime`) and `cell_id`.

        Args:
            agg_func:       Aggregation function to use for the main value,
                            e.g. 'mean', ...
            extra_stats:    Additional metrics to calculate. Provided as
                            {'suffix': 'statistic'}. Metric can be any
                            function or string that can be passed to
                            pandas.DataFrame.agg.

        Returns:
            statistics: An aggregated DataFrame with the calculated statistics.
                        Extra statistics will be added as new columns with the
                        name in the form of <metric>_<suffix>.

        Examples:
            Create a ForecastGRRateGrid from a dictionary with two grid cells,
            each having two realizations (as indicated by `grid_id`).

            .. code-block:: python

                >>> import pandas as pd
                >>> from seismostats import ForecastGRRateGrid
                >>> data = { 'longitude_min': [9, 9, 10, 10],
                ...          'longitude_max': [10, 10, 11, 11],
                ...          'latitude_min': [45, 45, 46, 46],
                ...          'latitude_max': [46, 46, 47, 47],
                ...          'depth_min': [10, 10, 20, 20],
                ...          'depth_max': [20, 20, 30, 30],
                ...          'number_events': [5, 6, 10, 12],
                ...          'a': [0.8, 0.9, 1.0, 1.1],
                ...          'b': [0.95, 1.0, 1.05, 1.1],
                ...          'mc': [1.2, 1.2, 1.3, 1.3],
                ...          'grid_id': [0, 1, 0, 1]}
                >>> forecast = ForecastGRRateGrid(
                ...     data,
                ...     starttime=pd.Timestamp('2023-01-01'),
                ...     endtime=pd.Timestamp('2023-01-02'))
                >>> forecast

                   longitude_min  longitude_max  ...    a     b   mc  grid_id
                0            9.0           10.0  ...  0.8  0.95  1.2        0
                1            9.0           10.0  ...  0.9  1.00  1.2        1
                2           10.0           11.0  ...  1.0  1.05  1.3        0
                3           10.0           11.0  ...  1.1  1.10  1.3        1

            Compute the mean and standard deviation of the GR parameters
            for each grid cell:

            .. code-block:: python

                >>> stats = forecast.calculate_statistics(
                ...     agg_func='mean',
                ...     extra_stats={'std': lambda x: x.std(ddof=0)}
                ... )
                >>> stats

                   longitude_min  longitude_max  ...      b   mc   b_std  mc_std
                0            9.0           10.0  ...  0.975  1.2  0.0353     0.0
                1           10.0           11.0  ...  1.075  1.3  0.0353     0.0

        """
        # Base columns to compute statistics on
        rategrid = self.copy()
        rategrid = rategrid.dropna(axis=1, how='all')

        # get available columns
        base_cols = ['number_events', 'b', 'mc', 'a', 'alpha']
        base_cols = [col for col in base_cols if col in rategrid.columns]
        numeric_cols = rategrid.select_dtypes(include='number').columns

        # Apply agg_func to all numeric columns
        agg_dict = {col: agg_func for col in numeric_cols}

        # Add extra metrics for base_cols (if not already present)
        for suffix, func in extra_stats.items():
            for col in base_cols:
                derived_col = f"{col}_{suffix}"
                if derived_col not in rategrid.columns:
                    rategrid[derived_col] = rategrid[col]
                    agg_dict[derived_col] = func  # Only add if new

        # Perform the groupby and aggregation
        if rategrid.index.nlevels == 1:
            agg = {'by': ['longitude_min', 'longitude_max',
                          'latitude_min', 'latitude_max',
                          'depth_min', 'depth_max']}
        else:  # aggregate by all levels
            agg = {'level': list(range(rategrid.index.nlevels))}

        statistics = rategrid.groupby(**agg).agg(agg_dict)

        if 'by' in agg:
            statistics.reset_index(inplace=True, drop=True)

        statistics.drop(columns=['grid_id'], inplace=True, errors='ignore')

        return statistics
