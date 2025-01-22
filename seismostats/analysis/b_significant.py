"""This module contains functions for the evaluation if a one dimensional
b-value series varies significantly, following Mirwald et al, SRL (2024)
"""

import numpy as np
import datetime as dt
import warnings

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.base import BValueEstimator


def est_morans_i(values: np.ndarray, w: np.ndarray | None):
    """
    Estimate the nearest neighbor auto correlation (Moran's I) of the values.

    Args:
        values:     values
        w:          Weight matrix, indicating which of the values are
                neighbors to each other. It should be a square matrix of
                size len(b_vec) x len(b_vec). At places where the value is nan,
                the matrix is set to zero, effectively not counting these data
                points. If w is None, it is assumed that the series of values is
                1-dimensional, and the values are sorted along that dimension.

    Returns:
        ac:     Auto correlation of the values
        n_p:    Sum of the weight matrix. In the limit of a large n (number of
            values), the upper limit of the standard deviation of the
            autocorrelation is 1/sqrt(n_p).

    """

    ac = 0
    ac_0 = 0

    # in case w is not provided, the 1-dimensional case is assumed
    if w is None:
        n_values = len(values)
        w = np.zeros(n_values, n_values)
        for ii in range(n_values):
            for jj in range(n_values):
                if ii == jj + 1:
                    w[ii, jj] = 1

    # estimate mean
    n = len(values[~np.isnan(values)])
    mean_v = np.mean(values[~np.isnan(values)])

    for ii, v1 in enumerate(values):
        if np.isnan(v1):
            w[ii, :] = 0
            continue
        ac_0 += (v1 - mean_v) ** 2
        for jj, v2 in enumerate(values):
            if np.isnan(v2):
                w[ii, jj] = 0
                continue
            if w[ii, jj] == 1:
                ac += (v1 - mean_v) * (v2 - mean_v)

    n_p = np.sum(w)
    ac = (n - 1) / n_p * ac / ac_0
    return ac, n, n_p


def transform_n(
    x: np.ndarray, b: float, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """transform b-value to be comparable to other b-values

    Args:
        x:  b-value estimates to be transformed
        b:  true b-value
        n1:   number of events used for the the b-value estimates
        n2:   number of events to which the distribution is transformed

    Returns:
        x:  transformed b-values
    """
    x_transformed = b / (1 - np.sqrt(n1 / n2) * (1 - b / x))
    return x_transformed


def b_samples(
    tile_magnitudes: np.ndarray,
    tile_times: np.ndarray[dt.datetime],
    delta_m: float,
    mc: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    return_std: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """ Estimate the b-values for a given partition of the data

    Args:
        list_mags:  list of arrays of magnitudes
        list_times: list of arrays of times
        delta_m:    discretization of magnitudes
        mc:         completeness magnitude (can be a vector of same
                length as list_magnitudes)
        b_method:   method to estimate the b-value
        return_std: if True, return the standard deviation of the b-values

    """

    b_series = np.zeros(len(tile_magnitudes))
    std_b = np.zeros(len(tile_magnitudes))
    n_ms = np.zeros(len(tile_magnitudes))
    if mc is float:
        mc = np.ones(len(tile_magnitudes)) * mc

    estimator = b_method()

    for ii, mags_loop in enumerate(tile_magnitudes):
        # sort the magnitudes of the subsets by time
        times_loop = tile_times[ii]
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]
        times_loop = times_loop[idx_sorted]

        if len(mags_loop) > 2:
            estimator.calculate(mags_loop, mc=mc[ii], delta_m=delta_m)
            b_series[ii] = estimator.b_value
            std_b[ii] = estimator.std
            n_ms[ii] = estimator.n
        else:
            b_series[ii] = np.nan
            std_b[ii] = np.nan

    if return_std:
        return b_series, std_b, n_ms.astype(int)

    return b_series, n_ms.astype(int)


def cut_constant_idx(
    series: np.ndarray,
    n_sample: np.ndarray | None = None,
    n_m: np.ndarray | None = None,
    offset: int = 0,
) -> tuple[list[int], list[np.ndarray]]:
    """cut a series such that the subsamples have a constant number of events.
    it is assumed that the magnitudes are ordered as desired (e.g. in time or
    in depth)

    Args:
        series:     array of values
        n_sample:   number of subsamples to cut the series into
        n_m:        length of each sample (if not given, it'll be estimated
                from n_sample)
        offset:     offset where to start cutting the series

    Returns:
        idx:            indices of the subsamples
        subsamples:     list of subsamples
    """
    if n_m is None:
        n_m = np.round(len(series) / n_sample).astype(int)
    idx = np.arange(offset, len(series), n_m)

    if offset == 0:
        idx = idx[1:]

    if offset > n_m:
        warnings.warn(
            "offset is larger than the number of events per subsample, this"
            "will lead to cutting off more events than necessary"
        )

    subsamples = np.array_split(series, idx)

    return idx, subsamples
