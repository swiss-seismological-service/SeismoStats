from __future__ import annotations

import pandas as pd

from catalog_tools.utils import _check_required_cols, require_cols

REQUIRED_COLS_RATEGRID = [
    'longitude_min', 'longitude_max',
    'latitude_min', 'latitude_max',
    'depth_min', 'depth_max',
    'number_events', 'a', 'b', 'mc'
]


def _rategrid_constructor_with_fallback(*args, **kwargs):
    df = RateGrid(*args, **kwargs)
    if not _check_required_cols(df, REQUIRED_COLS_RATEGRID):
        return pd.DataFrame(*args, **kwargs)
    if not _check_required_cols(df, required_cols=['grid_id']):
        return df
    return ForecastRateGrid(*args, **kwargs)


class RateGrid(pd.DataFrame):
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

    @property
    def _constructor(self):
        return _rategrid_constructor_with_fallback

    @require_cols(require=_required_cols)
    def strip(self, inplace: bool = False) -> RateGrid | None:
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


class ForecastRateGrid(RateGrid):
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
