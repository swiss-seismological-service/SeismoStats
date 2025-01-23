"""This module contains functions for the evaluation if a one dimensional
b-value series varies significantly, following Mirwald et al, SRL (2024)
"""

import numpy as np
import warnings

from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.base import BValueEstimator


def est_morans_i(values: np.ndarray, w: np.ndarray | None = None) -> tuple:
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
                Then, the ac that is returned corresponds to the usual 1D
                autocorrelation with a lag of 1.

    Returns:
        ac:     Auto correlation of the values
        n:      Number of values that are not nan
        n_p:    Sum of the weight matrix. In the limit of a large n (number of
            values), the upper limit of the standard deviation of the
            autocorrelation is 1/sqrt(n_p). This number is can be interpreted as
            the number of neighboring pairs.

    """

    ac = 0
    ac_0 = 0

    # in case w is not provided, the 1-dimensional case is assumed
    if w is None:
        n_values = len(values)
        w = np.zeros((n_values, n_values))
        for ii in range(n_values):
            for jj in range(n_values):
                if jj == ii + 1:
                    w[ii, jj] = 1
    else:
        if w.shape[0] != w.shape[1]:
            raise ValueError("Weight matrix must be square")
        if w.shape[0] != len(values):
            raise ValueError(
                "Weight matrix must have the same size as the values")
        # check that w is diagonal and has zeros on the diagonal
        if sum(w.diagonal()) != 0:
            raise ValueError("Weight matrix must have zeros on the diagonal")
        # check if w is triangular
        if np.sum(np.tril(w)) != 0 and np.sum(np.triu(w)) != 0:
            if np.all(w == w.T):
                w = np.triu(w)
            else:
                raise ValueError(
                    "Weight matrix must be triangular or at least symmetric")
        elif np.sum(np.triu(w)) == 0:
            w = w.T

        # estimate mean
    n = len(values[~np.isnan(values)])
    mean_v = np.mean(values[~np.isnan(values)])

    for ii, v1 in enumerate(values):
        if np.isnan(v1):
            w[ii, :] = 0
            continue
        ac_0 += (v1 - mean_v) ** 2
        for jj in range(ii + 1, len(values)):
            v2 = values[jj]
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


def b_series(
    list_magnitudes: list[np.ndarray],
    list_times: list[np.ndarray[np.datetime64]],
    delta_m: float,
    mc: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Estimate the series of b-values

    Args:
        list_mags:  list of arrays of magnitudes
        list_times: list of arrays of times
        delta_m:    discretization of magnitudes
        mc:         completeness magnitude (can be a vector of same
                length as list_magnitudes)
        b_method:   method to estimate the b-value

    Returns:
        b_values:   series of b-values, each one is estimated from the
            magnitudes contained in the corresponding element of list_magnitudes
        std_b:      standard deviations corresponding to the b-values
        n_ms:       number of events used for the b-value estimates
    """

    b_values = np.zeros(len(list_magnitudes))
    std_b = np.zeros(len(list_magnitudes))
    n_ms = np.zeros(len(list_magnitudes))
    if isinstance(mc, (float, int)):
        mc = np.ones(len(list_magnitudes)) * mc

    estimator = b_method()

    for ii, mags_loop in enumerate(list_magnitudes):
        # sort the magnitudes of the subsets by time
        times_loop = list_times[ii]
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]
        times_loop = times_loop[idx_sorted]

        try:
            estimator.calculate(mags_loop, mc=mc[ii], delta_m=delta_m)
            b_values[ii] = estimator.b_value
            std_b[ii] = estimator.std
            n_ms[ii] = estimator.n
        except Exception:
            b_values[ii] = np.nan
            std_b[ii] = np.nan

    return b_values, std_b, n_ms.astype(int)


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
    if n_sample is not None and n_m is not None:
        raise ValueError("Either n_sample or n_m must be given, not both")
    elif n_m is None:
        if n_sample is None:
            raise ValueError("Either n_sample or n_m must be given")
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
