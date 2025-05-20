import warnings
from typing import Literal

import numpy as np

from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def beta_to_b_value(beta: float) -> float:
    '''Converts the beta value to the b-value  of the Gutenberg-Richter law.

    Args:
        beta: beta value.

    Returns:
        b_value: Corresponding b-value.
    '''
    return beta / np.log(10)


def b_value_to_beta(b_value: float) -> float:
    '''Converts the b-value to the beta value of the exponential distribution.

    Args:
        b_value: b-value.

    Returns:
        beta: Corresponding beta value.
    '''
    return b_value * np.log(10)


def shi_bolt_confidence(
    magnitudes: np.ndarray,
    b: float,
    weights: np.ndarray | None = None,
    b_parameter: Literal['b_value', 'beta'] = 'b_value'
) -> float:
    '''
    Returns the Shi and Bolt (1982) confidence limit of the b-value or
    beta.

    Source:
        Shi and Bolt, BSSA, Vol. 72, No. 5, pp. 1677-1687, October 1982

    Args:
        magnitudes:     Array of magnitudes.
        weights:        Array of weights for the magnitudes.
        b:              Known or estimated b-value/beta of the magnitudes.
        b_parameter:    Either either 'b_value' or 'beta'.

    Returns:
        std_b:  Confidence limit of the b-value/beta value (depending on input).
    '''
    # standard deviation in Shi and Bolt is calculated with 1/(N*(N-1)), which
    # is by a factor of sqrt(N) different to the std(x, ddof=1) estimator
    if b_parameter not in ['b_value', 'beta']:
        raise ValueError('b_parameter must be either "b_value" or "beta"')

    std_mags = np.sqrt(np.average(np.square(
        magnitudes - np.average(magnitudes, weights=weights)), weights=weights))
    if weights is None:
        len_mags = len(magnitudes)
    else:
        len_mags = np.sum(weights)

    std_b = (
        np.log(10) * b**2 * std_mags / np.sqrt(len_mags - 1)
    )
    if b_parameter == 'beta':
        std_b = (std_b) / np.log(10)

    return std_b


def find_next_larger(magnitudes: np.array,
                     delta_m: float,
                     dmc: float | None):
    """
    Takes an array of magnitudes and returns an array of indices of the next
    largest event for each element. For each magnitude[ii], magnitude[idx[ii]]
    is the next larger event in the series. Example: magnitudes = [10, 4, 3, 9]
    result in [0, 3, 3, 0]. Note that the value of idx is 0 if no
    next magnitude exists.

    Args:
        magnitudes:     Array of magnitudes, ordered by a dimension of interest,
                    e.g., time).
        delta_m:        Bin size of discretized magnitudes.
        dmc:            Minimum magnitude difference between consecutive events.
                    If `None`, the default value is `delta_m`.

    """
    if dmc is None:
        dmc = delta_m

    idx_next_larger = np.zeros(len(magnitudes))
    for ii in range(len(magnitudes) - 1):
        for jj in range(ii + 1, len(magnitudes)):
            mag_diff_loop = magnitudes[jj] - magnitudes[ii]
            if mag_diff_loop >= dmc - delta_m / 2:
                idx_next_larger[ii] = jj
                break
    return idx_next_larger.astype(int)


def make_more_incomplete(
    magnitudes: np.ndarray,
    times: np.array,
    delta_t: np.timedelta64 = np.timedelta64(60, 's'),
    return_idx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns filtered magnitudes and times. Filters the magnitudes and times in
    the following way: If an earthquake is smaller than the previous one and
    less than ``delta_t`` away, the earthquake is removed.

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Args:
        magnitudes: Array of magnitudes, ordered in time (first
                entry is the earliest earthquake).
        times:      Array of datetime objects of occurrence of each earthquake.
        delta_t:    Time window in seconds to filter out events. Default is 60
                seconds.
        return_idx: If `True`, the indices of the events that were kept are
                also returned.

    Returns:
        magnitudes: Array of filtered magnitudes.
        times:      Array of filtered datetime objects.
        idx:        Array of indices of the events that were kept.
        '''

    # sort magnitudes in time
    idx_sort = np.argsort(times)
    magnitudes = magnitudes[idx_sort]
    times = times[idx_sort]

    idx = np.full(len(magnitudes), True)
    for ii in range(1, len(magnitudes)):
        # get all times that are closer than delta_t
        idx_close = np.where(times[ii] - times[:ii] < delta_t)[0]

        # check if these events are larger than the current event
        idx_loop = magnitudes[idx_close] > magnitudes[ii]

        # if there are any, remove the current event
        if sum(idx_loop) > 0:
            idx[ii] = False

    magnitudes = magnitudes[idx]
    times = times[idx]

    if return_idx is True:
        return magnitudes, times, idx

    return magnitudes, times


def cdf_discrete_exp(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cumulative distribution function (CDF) for a discrete
    exponential distribution at the points of the magnitudes.

    Args:
        magnitudes: Array of magnitudes.
        mc:         Completeness magnitude.
        delta_m:    Bin size of discretized magnitudes.
        beta:       Rate parameter of the exponential distribution.

    Returns:
        x: Unique x-values of the magnitudes.
        y: Corresponding y-values of the CDF of the GR distribution.
    """

    x = np.sort(magnitudes)
    x = np.unique(x)
    y = 1 - np.exp(-beta * (x + delta_m - mc))
    return x, y


def ks_test_gr(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float,
    b_value: float,
    n: int = 10000,
    ks_ds: list | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float, list[float]]:
    """
    Performs the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution for a given magnitude sample and mc and b-value. When the
    p-value is below a certain threshold (e.g., 0.1), the null hypothesis that
    the sample is drawn from a Gutenberg-Richter distribution with the given
    parameters can be rejected.

    Args:
        magnitudes: Array of magnitudes.
        mc:         Completeness magnitude.
        delta_m:    Bin size of discretized magnitudes.
        b_value:    b-value of the Gutenberg-Richter law.
        n:          Number of times the KS distance is calculated from
                synthetic samples with the given parameters, used for
                estimating the p-value.
        ks_ds:      KS distances from synthetic data with the given
                paramters. If None, they will be estimated here (then, n is
                not needed).
        weights:    Array of weights for the magnitudes.

    Returns:
        p_val:      p-value.
        ks_d_obs:   KS distance of the sample.
        ks_ds:      Array of KS distances.
    """
    if get_option("warnings") is True:
        if np.min(magnitudes) < mc - delta_m / 2:
            warnings.warn("Sample contains values below mc.")

        if len(magnitudes) == 0:
            warnings.warn("No sample given.")
            return 0, 1, []

        if len(np.unique(magnitudes)) == 1:
            warnings.warn("Sample contains only one value.")
            return 0, 1, []

    beta = b_value_to_beta(b_value)
    n_sample = len(magnitudes)

    if ks_ds is None:
        ks_ds = []

        # max considered magnitude is extrapolated from a bootstrap sample
        bootstrap_sample = simulate_magnitudes_binned(
            1000, b_value, mc, delta_m, b_parameter="b_value")
        max_simulated_mag = np.max(bootstrap_sample)
        safety_margin = max(np.log10(n / 1000) + 2, 1)
        max_considered_mag = max(
            np.max(magnitudes), max_simulated_mag + safety_margin)

        x_bins = bin_to_precision(
            np.arange(mc, max_considered_mag + 3
                      / 2 * delta_m, delta_m), delta_m
        )
        x = x_bins[:-1].copy()
        x_bins -= delta_m / 2
        _, y_th = cdf_discrete_exp(
            x, mc=mc, delta_m=delta_m, beta=beta)

        ks_ds = np.empty(n)
        for ii in range(n):
            simulated = simulate_magnitudes_binned(
                n_sample, b_value, mc, delta_m, b_parameter="b_value"
            )
            y_hist, _ = np.histogram(simulated, bins=x_bins, weights=weights)
            y_emp = np.cumsum(y_hist) / np.sum(y_hist)
            ks_ds[ii] = np.max(np.abs(y_emp - y_th))

    else:
        max_considered_mag = np.max(magnitudes)
        x_bins = bin_to_precision(
            np.arange(mc, max_considered_mag + 3
                      / 2 * delta_m, delta_m), delta_m
        )
        x = x_bins[:-1].copy()
        x_bins -= delta_m / 2
        _, y_th = cdf_discrete_exp(x, mc=mc, delta_m=delta_m, beta=beta)

    y_hist, _ = np.histogram(magnitudes, bins=x_bins, weights=weights)
    y_emp = np.cumsum(y_hist) / np.sum(y_hist)

    ks_d_obs = np.max(np.abs(y_emp - y_th))
    p_val = sum(ks_ds >= ks_d_obs) / len(ks_ds)

    return p_val, ks_d_obs, ks_ds
