from __future__ import annotations

import numpy as np
import pandas as pd

from catalog_tools.utils import _check_required_cols, require_cols

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
    A subclass of pandas DataFrame that represents a grid where for each
    grid cell, the GR parameters a, b, and mc and number_events are stored.

    To be a valid RateGrid object, the DataFrame must have the following
    columns: longitude_min, longitude_max, latitude_min, latitude_max,
    depth_min, depth_max, number_events, a, b, and mc.

    Args:
        data : array-like, Iterable, dict, or DataFrame, optional
            Data to initialize the catalog with.
        name : str, optional
            Name of the catalog.
        starttime : str or datetime-like, optional
            Start time of the catalog. If a string, it must be in a format
            that can be parsed by pandas.to_datetime.
        endtime : str or datetime-like, optional
            End time of the catalog. If a string, it must be in a format
            that can be parsed by pandas.to_datetime.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to pandas
            DataFrame constructor.

    Notes:
        The RateGrid class is a subclass of pandas DataFrame, and inherits
        all of its methods and attributes.
    """
    _metadata = ['name', '_required_cols', 'starttime', 'endtime']
    _required_cols = REQUIRED_COLS_RATEGRID

    def __init__(self, data=None, *args, name=None,
                 starttime=None,
                 endtime=None,
                 **kwargs):
        super().__init__(data, *args, **kwargs)

        self.name = name

        self.starttime = starttime if isinstance(
            starttime, pd.Timestamp) else pd.to_datetime(starttime)

        self.endtime = endtime if isinstance(
            endtime, pd.Timestamp) else pd.to_datetime(endtime)

        if len(self.index.names) > 1:
            try:
                self.reindex_cell_id()
            except AttributeError:
                pass

    @property
    def _constructor(self):
        return _rategrid_constructor_with_fallback

    @require_cols(require=_required_cols)
    def strip(self, inplace: bool = False) -> GRRateGrid | None:
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

    @require_cols(require=_required_cols)
    def add_time_index(self, endtime=True):
        """
        Create MultiIndex using starttime, optionally endtime and a cell
        number for each spatial block.

        Args:
            endtime : bool, optional
                If True, create MultiIndex with starttime and endtime.
                Otherwise, create MultiIndex with only starttime.

        Returns:
            RateGrid
        """
        if not getattr(self, 'starttime', None) or \
                not getattr(self, 'endtime', None):
            raise AttributeError(
                'starttime and endtime must be set to use this method')

        index = (self.starttime, self.endtime) if endtime else self.starttime
        names = ['starttime', 'endtime'] if endtime else ['starttime']

        # rename the index to cell_id, will be set in constructor
        self.index.name = 'cell_id'

        df = pd.concat({index: self}, names=names)

        # manually set the metadata attributes
        for arg in self._metadata:
            setattr(df, arg, getattr(self, arg))

        return df

    @require_cols(require=_required_cols)
    def reindex_cell_id(self):
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


class ForecastGRRateGrid(GRRateGrid):
    """
    A subclass of pandas DataFrame that represents a forecast on a grid where
    for each grid cell, the GR parameters a, b, and mc and number_events
    are stored.

    To be a valid RateGrid object, the DataFrame must have the following
    columns: longitude_min, longitude_max, latitude_min, latitude_max,
    depth_min, depth_max, number_events, a, b, mc, and grid_id.

    Args:
        data : array-like, Iterable, dict, or DataFrame, optional
            Data to initialize the catalog with.
        name : str, optional
            Name of the catalog.
        starttime : str or datetime-like, optional
            Start time of the catalog. If a string, it must be in a format
            that can be parsed by pandas.to_datetime.
        endtime : str or datetime-like, optional
            End time of the catalog. If a string, it must be in a format
            that can be parsed by pandas.to_datetime.
        *args, **kwargs : optional
            Additional arguments and keyword arguments to pass to pandas
            DataFrame constructor.

    Notes:
        The ForecastRateGrid class is a subclass of pandas DataFrame, and
        inherits all of its methods and attributes.
    """

    _required_cols = REQUIRED_COLS_RATEGRID + ['grid_id']
