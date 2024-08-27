"""This module contains functions
for the estimation of the completeness magnitude.
"""

import warnings
import decimal

import numpy as np
import pandas as pd

from seismostats.analysis.estimate_beta import estimate_b
from seismostats.utils.binning import bin_to_precision, get_fmd
from seismostats.utils.simulate_distributions import (
    simulate_magnitudes_binned,
)
from seismostats.utils._config import get_option


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
    mc: float = None,
    delta_m: float = 1e-16,
    weights: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical cumulative distribution function (CDF)
    from a sample.

    Parameters:
        sample:     Magnitude sample
        mc:         Completeness magnitude, if None, the minimum of the sample
                is used
        delta_m:    Magnitude bin size, by default 1e-16. Its recommended to
                use the value that the samples are rounded to.
        weights:    Sample weights, by default None

    Returns:
        x:          x-values of the empirical CDF (i.e. the unique vector of
                magnitudes from mc to the maximum magnitude in the sample,
                binned by delta_m)
        y:          y-values of the empirical CDF (i.e., the empirical
                frequency observed in the sample corresponding to the x-values)
    """

    try:
        sample = sample.values
    except BaseException:
        pass

    try:
        weights = weights.values
    except BaseException:
        pass

    if delta_m == 0:
        raise ValueError("delta_m has to be > 0")

    if get_option("warnings") is True:
        # check if binning is correct
        if not np.allclose(sample, bin_to_precision(sample, delta_m)):
            warnings.warn(
                "Magnitudes are not binned correctly. "
                "Test might fail because of this."
            )
        if delta_m == 1e-16:
            warnings.warn(
                "delta_m = 1e-16, this might lead to extended computation time")

    if mc is None:
        mc = np.min(sample)

    idx1 = np.argsort(sample)
    x = sample[idx1]
    x, y_count = np.unique(x, return_counts=True)

    # add empty bins
    for mag_bin in bin_to_precision(
        np.arange(mc, np.max(sample) + delta_m, delta_m), delta_m
    ):
        if mag_bin not in x:
            x = np.append(x, mag_bin)
            y_count = np.append(y_count, 0)
    idx2 = np.argsort(x)
    x = x[idx2]
    y_count = y_count[idx2]

    # estimate the CDF
    if weights is None:
        y = np.cumsum(y_count) / len(sample)
    else:
        weights_sorted = weights[idx1]
        y = np.cumsum(weights_sorted) / weights_sorted.sum()

        # make sure that y is zero if there are no samples in the first bins
        for ii, y_loop in enumerate(y_count):
            if y_loop > 0:
                break
        leading_zeros = np.zeros(ii)
        y = np.append(leading_zeros, y[np.cumsum(y_count[ii:]) - 1])

    return x, y


def ks_test_gr(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    beta: float,
    n: int = 10000,
    ks_ds: list | None = None,
) -> tuple[float, float, list[float]]:
    """
    For a given magnitude sample and mc and beta,
    perform the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution, to check if the sample
    could have been drawn from a GR distribution.

    Args:
        sample:     Magnitude sample
        mc:         Completeness magnitude
        delta_m:    Magnitude bin size
        beta :      Beta parameter for the Gutenberg-Richter distribution
        n:          Number of times the KS distance is calculated from
                synthetic samples with the given parameters, used for
                estimating the p-value. By default 10000.
        ks_ds:      List of KS distances from synthetic data with the given
                paramters. If None, they will be estimated here (then, n is
                not needed). By default None.

    Returns:
        p_val:      p-value
        ks_d_obs:   KS distance of the sample
        ks_ds:      list of KS distances
    """
    if get_option("warnings") is True:
        if np.min(sample) < mc - delta_m / 2:
            warnings.warn("sample contains values below mc - delta_m / 2")

        if len(sample) == 0:
            warnings.warn("no sample")
            return 0, 1, []

        if len(np.unique(sample)) == 1:
            warnings.warn("sample contains only one value")
            return 0, 1, []

    if ks_ds is None:
        ks_ds = []

        n_sample = len(sample)
        simulated_all = simulate_magnitudes_binned(
            n * n_sample, beta, mc, delta_m, b_parameter="beta"
        )

        for ii in range(n):
            simulated = simulated_all[n_sample * ii: n_sample * (ii + 1)]
            x, y_emp = empirical_cdf(simulated, mc, delta_m)
            _, y_th = cdf_discrete_GR(x, mc=mc, delta_m=delta_m, beta=beta)

            ks_d = np.max(np.abs(y_emp - y_th))
            ks_ds.append(ks_d)

    x, y_emp = empirical_cdf(sample, mc, delta_m)
    _, y_th = cdf_discrete_GR(x, mc=mc, delta_m=delta_m, beta=beta)

    ks_d_obs = np.max(np.abs(y_emp - y_th))
    p_val = sum(ks_ds >= ks_d_obs) / len(ks_ds)

    return p_val, ks_d_obs, ks_ds


def mc_ks(
    sample: np.ndarray,
    delta_m: float,
    mcs_test: np.ndarray | None = None,
    p_pass: float = 0.1,
    stop_when_passed: bool = True,
    verbose: bool = False,
    beta: float | None = None,
    b_method: str | None = None,
    n: int = 10000,
    ks_ds_list: list[list] | None = None,
) -> tuple[np.ndarray, list[float], np.ndarray, float | None, float | None]:
    """
    Return the completeness magnitude (mc) estimate
    for a given list of completeness
    magnitudes using the K-S distance method.

    Source:
        - Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
          distributions in empirical data. SIAM review, 51(4), pp.661-703.
        - Mizrahi, L., Nandan, S. and Wiemer, S., 2021. The effect of
          declustering on the size distribution of mainshocks. Seismological
          Society of America, 92(4), pp.2333-2342.

    Args:
        sample:             Magnitudes to test
        delta_m:            Magnitude bins (sample has to be rounded to bins
                            beforehand)
        mcs_test:           Completeness magnitudes to test, by default None
        p_pass:             P-value with which the test is passed, by default
                            0.1
        stop_when_passed:   Stop calculations when first mc passes the test, by
                            default True
        verbose:            Verbose output, by default False
        beta:               If beta is 'known', only estimate mc, by default
                            None
        n:                  Number of number of times the KS distance is
                            calculated for estimating the p-value,
                            by default 10000
        ks_ds_list:         List of list of KS distances from synthetic data
                        (needed for testing). If None, they will be estimated
                        in this funciton. By default None

    Returns:
        mcs_test:   tested completeness magnitudes
        ks_ds:      KS distances
        ps:         p-values
        best_mc:    best mc
        beta:       corresponding best beta
    """

    if mcs_test is None:
        mcs_test = bin_to_precision(
            np.arange(np.min(sample), np.max(sample), delta_m), delta_m
        )
    else:
        # check that they are ordered by size
        if get_option("warnings") is True:
            if not np.all(np.diff(mcs_test) > 0):
                warnings.warn("mcs_test are being re-ordered by size.")
                mcs_test = np.sort(np.unique(mcs_test))
            if not np.allclose(mcs_test, bin_to_precision(mcs_test, delta_m)):
                warnings.warn(
                    "mc_test are not binned correctly,"
                    "this might affect the test."
                )

    if get_option("warnings") is True:
        # check if binning is correct
        if not np.allclose(sample, bin_to_precision(sample, delta_m)):
            warnings.warn(
                "Magnitudes are not binned correctly. "
                "Test might fail because of this."
            )

        if not np.allclose(mcs_test, bin_to_precision(mcs_test, delta_m)):
            warnings.warn(
                "Mcs to test are not binned correctly. "
                "Test might fail because of this."
            )

        # check if beta is given (then b_method is not needed)
        if beta is not None and b_method is not None:
            warnings.warn("Both beta and b_method are given. Using beta.")

    if beta is None and b_method is None:
        b_method = "classic"

    mcs_tested = []
    ks_ds = []
    ps = []
    betas = []

    for ii, mc in enumerate(mcs_test):

        if verbose:
            print("\ntesting mc", mc)

        mc_sample = sample[sample >= mc - delta_m / 2]

        # if no beta is given, estimate beta
        if beta is None:
            mc_beta = estimate_b(
                magnitudes=mc_sample,
                mc=mc,
                delta_m=delta_m,
                b_parameter="beta",
                method=b_method,
            )
        else:
            mc_beta = beta

        if ks_ds_list is None:
            p, ks_d, _ = ks_test_gr(
                mc_sample, mc=mc, delta_m=delta_m, beta=mc_beta, n=n
            )
        else:
            p, ks_d, _ = ks_test_gr(
                mc_sample,
                mc=mc,
                delta_m=delta_m,
                beta=mc_beta,
                n=n,
                ks_ds=ks_ds_list[ii],
            )

        mcs_tested.append(mc)
        ks_ds.append(ks_d)
        ps.append(p)
        betas.append(mc_beta)

        if verbose:
            print("..p-value: ", p)

        if p >= p_pass and stop_when_passed:
            break

    ps = np.array(ps)

    if np.any(ps >= p_pass):
        best_mc = mcs_tested[np.argmax(ps >= p_pass)]
        best_beta = betas[np.argmax(ps >= p_pass)]

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

    return best_mc, best_beta, mcs_tested, betas, ks_ds, ps


def mc_max_curvature(
    sample: np.ndarray,
    delta_m: float,
    correction_factor: float = 0.2,
) -> float:
    """
    Return the completeness magnitude (mc) estimate
    using the maximum curvature method.

    Source:
        - Wiemer, S. and Wyss, M., 2000. Minimum magnitude of completeness
          in earthquake catalogs: Examples from Alaska, the western United
          States, and Japan. Bulletin of the Seismological Society of America,
          90(4), pp.859-869.
        - Woessner, J. and Wiemer, S., 2005. Assessing the quality of earthquake
          catalogues: Estimating the magnitude of completeness and its
          uncertainty.
          Bulletin of the Seismological Society of America, 95(2), pp.684-698.

    Args:
        sample:     Magnitudes to test
        delta_m:    Magnitude bins (sample has to be rounded to bins beforehand)
            correction_factor:  Correction factor for the maximum curvature
            method (default 0.2 after Woessner & Wiemer 2005)

    Returns:
        mc:                 estimated completeness magnitude
    """
    bins, count, mags = get_fmd(
        mags=sample, delta_m=delta_m, bin_position="center"
    )
    mc = bins[count.argmax()] + correction_factor
    return mc


def mc_by_bvalue_stability(
        sample: np.ndarray, delta_m: float, stability_range: float = 0.5):
    """
    Estimates Mc using a test of stability.
    The stability of the b-value is tested by default on half a magnitude unit
    (in line with the 5x0.1 in the orginial paper). Users can change the range
    for the stability test by changing the stability_range.

    Source:
        Cao, A., & Gao, S. S. (2002). Temporal variation of seismic b-values
            beneath northeastern Japan island arc. Geophysical Research
            Letters, 29(9), 1–3. https://doi.org/10.1029/2001gl013775

    Args:
        sample : np.array Vector of magnitudes
        delta_m : float. Discretization of the magnitudes.
        stability_range : float. Magnitude range to consider for the
            stability test. Default is 0.5 to consider half a magnitude unit,
            this is compatible with the original definition of Cao & Gao 2002.

    Returns:
        mcs_test : np.ndarray. Tested completeness magnitudes
        mc : float. Single best magnitude of completeness estimate
        b : float. b-value associated with best_mc
        n : int. Number of events greater than or equal to best_mc
        sb_b_err : np.ndarray. Standard error of the b-value for all tested Mc
        diff_b : np.ndarray. Difference between b-estimate and b-avg for each Mc
        b_avgs : np.ndarray. Average b-value looking forward over
            stability_range for each step in Mc
        bs : np.ndarray. Estimated b-value for each step in Mc
        """
    sample = bin_to_precision(sample, delta_x=delta_m)
    # Define mc_span
    mcs_test = bin_to_precision(np.arange(np.min(sample),
                                          np.max(sample),
                                          delta_m), delta_m)
    mcs_test = mcs_test[:-4]
    bs = []
    sb_b_err = []
    b_avgs = []

    # bin sample to precision
    sample = bin_to_precision(sample, delta_m)

    for mc in mcs_test:
        b, err = estimate_b(sample[sample >= mc - delta_m / 2], mc, delta_m,
                            b_parameter='b_value', return_std=True,
                            method="classic")
        # raise warning if number of events above Mc is less than 30
        if len(sample[sample >= mc]) < 30:
            warnings.warn(
                "Number of events above tested Mc is less than 30. "
                "This might affect the stability test."
            )
        bs.append(b)
        sb_b_err.append(err)
        discretisation = 10**- \
            abs(decimal.Decimal(str(stability_range)).as_tuple().exponent)
        mc_plus = bin_to_precision(np.arange(mc, bin_to_precision(
            mc + stability_range, delta_m), discretisation), delta_m)
        b_ex = []
        # truncate mc_plus to remove all values larger than the maximum
        # magnitude in the sample
        mc_plus = mc_plus[mc_plus <= np.max(sample)]

        for mcp in mc_plus:
            b = estimate_b(sample[sample >= mcp - delta_m / 2],
                           mcp, delta_m, b_parameter='b_value', method="classic")
            b_ex.append(b)
        b_avg = np.sum(b_ex) / 5
        b_avgs.append(b_avg)
    bs, sb_b_err, b_avgs = np.array(bs), np.array(sb_b_err), np.array(b_avgs)
    diff_b = np.abs(b_avgs - bs)

    # Select Mc
    for i, mc in enumerate(mcs_test):
        if diff_b[i] <= sb_b_err[i]:
            mc_best = mc
            b_best = bs[i]
            n = len(sample[sample >= mc_best])
            value = True
            break
        else:
            value = False

    if value:
        return mcs_test, mc_best, b_best, n, sb_b_err, diff_b, b_avgs, bs
    else:
        raise ValueError("No Mc passes the stability test")
