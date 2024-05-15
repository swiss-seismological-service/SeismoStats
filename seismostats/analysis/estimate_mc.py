"""This module contains functions
for the estimation of the completeness magnitude.
"""

import numpy as np
import pandas as pd

from seismostats.analysis.estimate_beta import estimate_b_tinti
from seismostats.utils.binning import get_fmd
from seismostats.utils.simulate_distributions import (
    simulated_magnitudes_binned,
)


def cdf_discrete_GR(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the cumulative distribution function (CDF)
    for a discrete Gutenberg-Richter distribution at the points of the sample.

    Parameters:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bins
        beta:       Beta parameter for the Gutenberg-Richter distribution

    Returns:
        x: unique x-values of the sample
        y: corresponding y-values of the CDF of the GR distribution
    """

    x = np.sort(sample)
    x = np.unique(x)
    y = 1 - np.exp(-beta * (x + delta_m - mc))
    return x, y


def empirical_cdf(
    sample: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cumulative distribution function (CDF)
    from a sample.

    Parameters:
        sample:     Magnitude sample
        weights:    Sample weights, by default None

    Returns:
        x:          x-values of the empirical CDF
        y:          y-values of the empirical CDF
    """

    try:
        sample = sample.values
    except BaseException:
        pass

    try:
        weights = weights.values
    except BaseException:
        pass

    idx = np.argsort(sample)
    sample_sorted = sample[idx]

    if weights is not None:
        weights_sorted = weights[idx]
        x, y = sample_sorted, np.cumsum(weights_sorted) / weights_sorted.sum()
    else:
        x, y = sample_sorted, np.arange(1, len(sample) + 1) / len(sample)

    x, y_count = np.unique(x, return_counts=True)
    y = y[np.cumsum(y_count) - 1]
    return x, y


def ks_test_gr(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    n: int = 10000,
    beta: float | None = None,
) -> tuple[float, float, list[float]]:
    """
    For a given magnitude sample and mc,
    perform the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution.

    Args:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bin size
        n:          Number of number of times the KS distance is calculated for
                estimating the p-value, by default 10000
        beta :      Beta parameter for the Gutenberg-Richter distribution, by
                    default None

    Returns:
        orig_ks_d:  original KS distance
        p_val:      p-value
        ks_ds:      list of KS distances
    """

    sample = sample[sample >= mc - delta_m / 2]

    if len(sample) == 0:
        print("no sample")
        return 1, 0, []

    if len(np.unique(sample)) == 1:
        print("sample contains only one value")
        return 1, 0, []

    if beta is None:
        beta = estimate_b_tinti(sample, mc=mc, delta_m=delta_m, b_parameter="beta")

    ks_ds = []

    n_sample = len(sample)
    simulated_all = simulated_magnitudes_binned(
        n * n_sample, beta, mc, delta_m, b_parameter="beta"
    )

    for ii in range(n):
        simulated = simulated_all[n_sample * ii : n_sample * (ii + 1)]
        _, y_th = cdf_discrete_GR(simulated, mc=mc, delta_m=delta_m, beta=beta)
        _, y_emp = empirical_cdf(simulated)

        ks_d = np.max(np.abs(y_emp - y_th))
        ks_ds.append(ks_d)

    _, y_th = cdf_discrete_GR(sample, mc=mc, delta_m=delta_m, beta=beta)
    _, y_emp = empirical_cdf(sample)

    orig_ks_d = np.max(np.abs(y_emp - y_th))
    p_val = sum(ks_ds >= orig_ks_d) / len(ks_ds)

    return orig_ks_d, p_val, ks_ds


def mc_ks(
    sample: np.ndarray,
    mcs_test: np.ndarray,
    delta_m: float,
    p_pass: float,
    stop_when_passed: bool = True,
    verbose: bool = False,
    beta: float | None = None,
    n: int = 10000,
) -> tuple[np.ndarray, list[float], np.ndarray, float | None, float | None]:
    """
    Estimate the completeness magnitude (mc) for a given list of completeness
    magnitudes using the K-S distance method.

    Args:
        sample:             Magnitudes to test
        mcs_test:           Completeness magnitudes to test
        delta_m:            Magnitude bins (sample has to be rounded to bins
                            beforehand)
        p_pass:             P-value with which the test is passed
        stop_when_passed:   Stop calculations when first mc passes the test, by
                            default True
        verbose:            Verbose output, by default False
        beta:               If beta is 'known', only estimate mc, by default
                            None
         n:                 Number of number of times the KS distance is
                        calculated for estimating the p-value, by default 10000
    Returns:
        mcs_test:   tested completeness magnitudes
        ks_ds:      KS distances
        ps:         p-values
        best_mc:    best mc
        beta:       corresponding best beta
    """

    ks_ds = []
    ps = []
    i = 0

    for mc in mcs_test:
        if verbose:
            print("\ntesting mc", mc)

        ks_d, p, _ = ks_test_gr(sample, mc=mc, delta_m=delta_m, n=n, beta=beta)

        ks_ds.append(ks_d)
        ps.append(p)

        i += 1

        if verbose:
            print("..p-value: ", p)

        if p >= p_pass and stop_when_passed:
            break

    ps = np.array(ps)

    if np.any(ps >= p_pass):
        best_mc = mcs_test[np.argmax(ps >= p_pass)]

        if beta is None:
            beta = estimate_b_tinti(
                sample[sample >= best_mc - delta_m / 2],
                mc=best_mc,
                delta_m=delta_m,
                b_parameter="beta",
            )

        if verbose:
            print(
                "\n\nFirst mc to pass the test:",
                best_mc,
                "\nwith a beta of:",
                beta,
            )
    else:
        best_mc = None
        beta = None

        if verbose:
            print("None of the mcs passed the test.")

    return mcs_test, ks_ds, ps, best_mc, beta


def mc_max_curvature(
    sample: np.ndarray,
    delta_m: float,
    correction_factor: float = 0.2,
) -> float:
    """
    Estimate the completeness magnitude (mc) by maximum curvature (Wiemer and
    Wyss 2000, Woessner and Wiemer 2005).
    Args:
        sample:             Magnitudes to test
        delta_m:            Magnitude bins (sample has to be rounded to bins
                            beforehand)
        correction_factor:  Correction factor for the maximum curvature method
        (default 0.2 after Woessner & Wiemer 2005)
    Returns:
        mc:                 estimated completeness magnitude
    """
    bins, count, mags = get_fmd(mags=sample, delta_m=delta_m, bin_position="center")
    mc = bins[count.argmax()] + correction_factor
    return mc
