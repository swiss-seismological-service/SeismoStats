"""This module contains functions for the evaluation if a one dimensional
b-value series varies significantly, following Mirwald et al, SRL (2024)
"""

import warnings

import numpy as np
from scipy.stats import norm

from seismostats.analysis.avalue import AValueEstimator
from seismostats.analysis.bvalue import ClassicBValueEstimator
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.utils._config import get_option


def est_morans_i(values: np.ndarray,
                 w: np.ndarray | None = None,
                 mean_v: float | float | None = None) -> tuple:
    """
    Estimates the nearest neighbor auto correlation (Moran's I) of the values.

    Args:
        values:     Values for which the autocorrelation is estimated.
        w:          neighbor matrix, indicating which of the values are
                neighbors to each other. It should be a square matrix of size
                :code:`len(values) x len(values)`, with zeros on the diagonal.
                At places where the value is 1, the values are considered
                neighbors to each other. Values that are NaN are not considered
                neighbors to any other value. If w is None, it is assumed that
                the series of values is 1-dimensional, and the values are
                sorted along that dimension. Then, the ac that is returned
                corresponds to the  usual 1D autocorrelation with a lag of 1.
        mean_v:     Mean value of the series. If not provided, it is estimated
                from the non-nan values.

    Returns:
        ac:         Auto correlation of the values.
        n:          Number of values that are not NaN.
        n_p:        Sum of the nearest neighbor matrix. In the limit of a large
                n (number of values), the upper limit of the standard deviation
                of the autocorrelation is `1/sqrt(n_p)`. This number is can be
                interpreted as the number of neighboring pairs.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis.b_significant import est_morans_i

            >>> values = np.array([2, 2, 2, 4, 4])
            >>> w = np.array([[0, 1, 1, 0, 0],
            ...             [1, 0, 1, 0, 0],
            ...             [1, 1, 0, 0, 0],
            ...             [0, 0, 0, 0, 1],
            ...             [0, 0, 0, 1, 0]])
            >>> ac, n, n_p = est_morans_i(values, w, mean_v=3)
            >>> ac
            0.8
    """
    # sanity checks
    if len(values) < 2:
        raise ValueError("At least 2 values are needed for the estimation.")
    values = np.array(values)

    # Checks regardning the nearest neigbor matrix. In case it is not provided,
    # the 1D case is assumed
    if w is None:
        n_values = len(values)
        w = np.eye(n_values, k=1)
    else:
        if w.shape[0] != w.shape[1]:
            raise ValueError("Neighbor matrix must be square.")
        if sum(w.diagonal()) != 0:
            np.fill_diagonal(w, 0)
            if get_option('warnings') is True:
                warnings.warn("Diagonal of the neighbour matrix is not zero."
                              "It is set to zero.")
        if np.sum(np.tril(w)) != 0 and np.sum(np.triu(w)) != 0:
            if np.all(w == w.T):
                w = np.triu(w)
            else:
                raise ValueError(
                    "Neighbor matrix must be triangular or symmetric.")
        elif np.sum(np.triu(w)) == 0:
            w = w.T
        if not np.all(np.isin(w, [0, 1])):
            raise ValueError(
                "Neighbor matrix must only contain the values 0 and 1.")

    # estimate autocorrelation
    valid_mask = ~np.isnan(values)
    if mean_v is None:
        mean_v = np.mean(values[valid_mask])

    w[~valid_mask, :] = 0
    w[:, ~valid_mask] = 0

    deviations = values - mean_v
    deviations[~valid_mask] = 0
    ac = deviations @ w @ deviations
    ac_0 = deviations @ deviations

    n_p = np.sum(w)
    n = sum(valid_mask)
    ac = (n - 1) / n_p * ac / ac_0
    return ac, n, n_p


def transform_n(
    b_estimate: float,
    b_true: float,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Transforms a b-value estimated from n1 events to a b-value estimated from
    n2 events, such that the distribution of the transformed b-values is
    consistent with one that would be estimated from n2 events. The
    transformation is based on the assumption that the true b-value is known,
    and that the b-values estimated follow the reciprocaln inverse distribution
    (which is only true for a large enough n1, see Shi and Bolt, 1981, BSSA).

    Source:
        Mirwald et al, SRL (2024), supplementary material

    Args:
        b_estimates:    b-value estimates to be transformed.
        b_true:         True b-value.
        n1:             Number of events used for the the b-value estimates.
        n2:             number of events to which the distribution is
                    transformed. Note that b_estimate, n1 and n2 can also be
                    arrays of the same length.

    Returns:
        b_transformed:  Transformed b-values.
    """

    # sanity checks
    if np.any(n1 > n2):
        raise ValueError("n2 must be larger or equal than n1.")

    # transform the b-values
    b_transformed = b_true / (1 - np.sqrt(n1 / n2) * (1 - b_true / b_estimate))
    return b_transformed


def values_from_partitioning(
    list_magnitudes: list[np.ndarray],
    list_times: list[np.ndarray],
    list_mc: list[float] | float,
    delta_m: float,
    method: AValueEstimator | BValueEstimator = ClassicBValueEstimator,
    list_scaling: list | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Estimates the series of b-values from a list of subsets of magnitudes
    and times.

    Args:
        list_mags:  List of arrays of magnitudes. From each array within the
                list, a b-value is estimated.
        list_times: List of arrays of times, in the same order as the magnitudes
        list_mc:    List of completeness magnitude corresponding to the
                magnitudes. If a single value is provided, it is used for all
                magnitudes.
        delta_m:    Bin size of discretized magnitudes.
        method:     AValueEstimator or BValueEstimator class to use for
                calculation.
        list_scaling: List of scaling factors for the a-value estimation. Only
                used in case the method is an a-value estimator.
        **kwargs:   Additional parameters to be passed to the a/b-value
                estimation method.

    Returns:
        values:     Series of b-values (or a-values), each one is estimated
                from the magnitudes contained in the corresponding element of
                ``list_magnitudes``.
        std_b:      Standard deviations corresponding to the b-values.
        n_ms:       Number of events used for the b-value estimates.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis.b_significant import (
            ...     values_from_partitioning)
            >>> mags = [np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            ...         np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
            ...         np.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 40])]
            >>> times = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ...         np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            ...         np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30])]
            >>> delta_m = 1
            >>> mc = 11
            >>> b_values, std_bs, n_ms = values_from_partitioning(
            ...     mags, times, delta_m, mc)
            >>> b_values
            array([0.0289637 , 0.0173741 , 0.01240926])
    """
    # sanity checks
    n_subsets = len(list_magnitudes)
    if n_subsets != len(list_times):
        raise IndexError(
            "Length of list_times and list_magnitudes must be the same.")
    list_magnitudes = [np.array(mags) for mags in list_magnitudes]
    list_times = [np.array(times) for times in list_times]
    if isinstance(list_mc, (float, int)):
        list_mc = np.ones(n_subsets) * list_mc
    else:
        if n_subsets != len(list_mc):
            raise IndexError(
                "Length of list_mc must be the same as list_magnitudes.")
        list_mc = np.array(list_mc)
    if issubclass(method, AValueEstimator):
        if list_scaling is None:
            list_scaling = np.ones(n_subsets)
        list_scaling = np.array(list_scaling)

    # start estimation
    estimator = method()

    values = np.zeros(n_subsets)
    stds = np.zeros(n_subsets)
    n_ms = np.zeros(n_subsets)
    for ii, mags_loop in enumerate(list_magnitudes):
        # Sort the magnitudes of the subsets by time.
        times_loop = list_times[ii]
        idx_sorted = np.argsort(times_loop)
        mags_loop = mags_loop[idx_sorted]
        times_loop = times_loop[idx_sorted]

        if isinstance(estimator, AValueEstimator):
            estimator.calculate(
                mags_loop,
                mc=list_mc[ii],
                delta_m=delta_m,
                scaling_factor=list_scaling[ii],
                **kwargs)
            values[ii] = estimator.a_value
        elif isinstance(estimator, BValueEstimator):
            estimator.calculate(
                mags_loop, mc=list_mc[ii], delta_m=delta_m, **kwargs)
            values[ii] = estimator.b_value
        stds[ii] = estimator.std
        n_ms[ii] = estimator.n

    return values, stds, n_ms.astype(int)


def cut_constant_idx(
    values: np.ndarray,
    n: int,
    offset: int = 0,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Finds the indices to cut a series such that the subsamples have a constant
    number of events, n.

    Args:
        values:     Original series of values to be cut.
        n:          Number of events in each subsample.
        offset:     Index where to start cutting the series. This should be
                    between 0 and n.

    Returns:
        idx:        Indices of the subsamples, which can be used to
                    construct the subsamples in the following way:
                    :code:`subsamples = np.array_split(values, idx)`
        subsamples: The subsamples.
    """
    # sanity checks
    if offset >= n:
        raise ValueError("Offset must be smaller than n.")
    values = np.array(values)

    idx = np.arange(offset, len(values), n)
    if offset == 0:
        idx = idx[1:]
    subsamples = np.array_split(values, idx)
    return idx, subsamples


def b_significant_1D(
        magnitudes: np.ndarray,
        mc: float | np.ndarray,
        delta_m: float,
        times: np.ndarray,
        n_m: int,
        min_num: int = 10,
        method: BValueEstimator | AValueEstimator = ClassicBValueEstimator,
        conservative: bool = True,
        ** kwargs,
) -> tuple[float, float, float, float]:
    """
    Estimates the significance of variation of b-values (or a-values) along a
    one-dimensional series of events.

    The function outputs the p-value of the null hypothesis that the true
    b-value (a-value) is constant, together with the mean autocorrelation (MAC)
    and its mean and standard deviation. The method is based on the assumption
    the MAC is normally distributed under H0, which is true for a large
    enough number of subsamples. As a lower limit, no less than 25 subsamples
    (which can be estimated by len(magnitudes) / n_m) should be used.

    Args:
        magnitudes:     Array of magnitudes of the events. They are assumed to
                    be ordered along the dimension of interest (e.g. time or
                    depth).
        mc:             Completeness magnitude. If a single value is provided,
                    it is used for all magnitudes. Otherwise, the individual
                    completeness of each magnitude can be provided. This will
                    be used to filter the magnitudes.
        delta_m:        Bin size of discretized magnitudes.
        times:          Array of times of the events.
        n_m:            Number of magnitudes in each partition.
        min_num:        Minimum number of events in a partition.
        method:         AValueEstimator or BValueEstimator class to use for
                    calculation.
        conservative:   If True, the conservative estimate of the standard
                    deviation of the autocorrelation is used, i.e., gamma = 1.
                    If False (default), the non-conservative estimate is used,
                    i.e., gamma = 0.81 (see Mirwald et al., SRL (2024)).
        **kwargs:       Additional parameters to be passed to the b-value
                    estimation method.

    Returns:
        p_value:    p-value of the null hypothesis that the b-values are
                constant.
        mac:        Mean autocorrelation.
        mu_mac:     Expected mean autocorrelation under H0.
        std_mac:    Standard deviation of the mean autocorrelation under H0.
                (i.e., constant b-value). Here, the conservatice estimate is
                used - in case the non-conservative estimate is needed, the
                standard deviation can be mulitplied by the factor gamma = 0.81
                given by Mirwald et al., SRL (2024).

    See Also:
        To plot the time series, use
        :func:`seismostats.plots.plot_b_series_constant_nm`. To plot the mean
        autocorrelation for different n_m, use
        :func:`seismostats.plots.plot_b_significant_1D`
    """
    # sanity checks and preparation
    magnitudes = np.array(magnitudes)
    times = np.array(times)
    if len(magnitudes) != len(times):
        raise IndexError("Magnitudes and times must have the same length.")
    if isinstance(mc, (float, int)):
        mc = np.ones(len(magnitudes)) * mc
    else:
        mc = np.array(mc)
    if n_m < min_num:
        raise ValueError("n_m cannot be smaller than min_num.")

    idx = magnitudes >= mc - delta_m / 2
    magnitudes = magnitudes[idx]
    times = times[idx]
    mc = mc[idx]

    if len(magnitudes) / n_m < 3:
        if get_option("warnings") is True:
            warnings.warn(
                "n_m is too large - less than three subsamples are created,"
                "returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan
    elif len(magnitudes) / n_m < 25:
        if get_option("warnings") is True:
            warnings.warn(
                "The number of subsamples is less than 25. The normality "
                "assumption of the autocorrelation might not be valid.")

    # Estimate a and b values for n_m realizations.
    ac_1D = np.zeros(n_m)
    n = np.zeros(n_m)
    n_p = np.zeros(n_m)
    n_ms = np.zeros(n_m)
    for ii in range(n_m):
        # partition data
        idx, list_magnitudes = cut_constant_idx(
            magnitudes, n_m, offset=ii
        )
        list_times = np.array_split(times, idx)
        list_mc = np.array_split(mc, idx)
        list_mc = [float(max(mc_loop)) for mc_loop in list_mc]

        # Make sure that data at the edges is not included if not enough
        # samples.
        if len(list_magnitudes[-1]) < n_m:
            list_magnitudes.pop(-1)
            list_times.pop(-1)
            list_mc.pop(-1)
        if len(list_magnitudes[0]) < n_m:
            list_magnitudes.pop(0)
            list_times.pop(0)
            list_mc.pop(0)

        # Estimate b-values (a-values)
        vec, _, n_m_loop = values_from_partitioning(
            list_magnitudes, list_times, list_mc,
            delta_m, method=method, **kwargs)
        vec[n_m_loop < min_num] = np.nan

        # Estimate average events per b-value estimate.
        n_ms[ii] = np.mean(n_m_loop[n_m_loop >= min_num])
        # estimate autocorrelation (1D)
        ac_1D[ii], n[ii], n_p[ii], = est_morans_i(vec)

    # Estimate mean and (conservative) standard deviation of the
    # autocorrelation under H0.
    mac = np.nanmean(ac_1D)
    mean_n = np.nanmean(n)
    mean_np = np.nanmean(n_p)
    mu_mac = -1 / mean_n
    std_mac = (mean_np - 2) / (mean_np * np.sqrt(mean_np))

    if not conservative:
        std_mac *= 0.81

    p = 1 - norm(loc=mu_mac, scale=std_mac).cdf(mac)
    return p, mac, mu_mac, std_mac
