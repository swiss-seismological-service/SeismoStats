"""This module contains functions
for the estimation of the completeness magnitude.
"""
import numpy as np
import pandas as pd

from seismostats.analysis.estimate_beta import estimate_b_tinti
from seismostats.utils.binning import normal_round
from seismostats.utils.simulate_distributions import simulate_magnitudes


def fitted_cdf_discrete(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    x_max: float | None = None,
    beta: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the fitted cumulative distribution function (CDF)
    for a discrete Gutenberg-Richter distribution.

    Parameters:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bins
        x_max:      Maximum magnitude, by default None
        beta:       Beta parameter for the Gutenberg-Richter distribution, by
                    default None

    Returns:
        x: x-values of the fitted CDF
        y: y-values of the fitted CDF
    """

    if beta is None:
        beta = estimate_b_tinti(
            sample, mc=mc, delta_m=delta_m, b_parameter="beta"
        )

    if x_max is None:
        sample_bin_n = (sample.max() - mc) / delta_m
    else:
        sample_bin_n = (x_max - mc) / delta_m

    bins = np.arange(sample_bin_n + 1)
    cdf = 1 - np.exp(-beta * delta_m * (bins + 1))
    x, y = mc + bins * delta_m, cdf

    x, y_count = np.unique(x, return_counts=True)
    y = y[np.cumsum(y_count) - 1]
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

    sample_idxs_sorted = np.argsort(sample)
    sample_sorted = sample[sample_idxs_sorted]

    if weights is not None:
        weights_sorted = weights[sample_idxs_sorted]
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
    ks_ds: list[float] | None = None,
    n_samples: int = 10000,
    beta: float | None = None,
) -> tuple[float, float, list[float]]:
    """
    Perform the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution.

    Args:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bin size
        ks_ds:      List to store KS distances, by default None
        n_samples:  Number of magnitude samples to be generated in p-value
                    calculation of KS distance, by default 10000
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
        beta = estimate_b_tinti(
            sample, mc=mc, delta_m=delta_m, b_parameter="beta"
        )

    if ks_ds is None:
        ks_ds = []

        n_sample = len(sample)
        simulated_all = (
            normal_round(
                simulate_magnitudes(
                    mc=mc - delta_m / 2, beta=beta, n=n_samples * n_sample
                )
                / delta_m
            )
            * delta_m
        )

        x_max = np.max(simulated_all)
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, x_max=x_max, beta=beta
        )

        for i in range(n_samples):
            simulated = simulated_all[n_sample * i : n_sample * (i + 1)].copy()
            x_emp, y_emp = empirical_cdf(simulated)
            y_fit_int = np.interp(x_emp, x_fit, y_fit)

            ks_d = np.max(np.abs(y_emp - y_fit_int))
            ks_ds.append(ks_d)
    else:
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, beta=beta
        )

    x_emp, y_emp = empirical_cdf(sample)
    y_emp_int = np.interp(x_fit, x_emp, y_emp)

    orig_ks_d = np.max(np.abs(y_fit - y_emp_int))
    p_val = sum(ks_ds >= orig_ks_d) / len(ks_ds)

    return orig_ks_d, p_val, ks_ds


def estimate_mc(
    sample: np.ndarray,
    mcs_test: np.ndarray,
    delta_m: float,
    p_pass: float,
    stop_when_passed: bool = True,
    verbose: bool = False,
    beta: float | None = None,
    n_samples: int = 10000,
) -> tuple[np.ndarray, list[float], np.ndarray, float | None, float | None]:
    """
    Estimate the completeness magnitude (mc) for a given list of completeness
    magnitudes.

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
        n_samples:          Number of magnitude samples to be generated in
                            p-value calculation of KS distance, by default 10000

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

        ks_d, p, _ = ks_test_gr(
            sample, mc=mc, delta_m=delta_m, n_samples=n_samples, beta=beta
        )

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
