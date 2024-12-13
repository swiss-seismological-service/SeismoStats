"""This module contains functions
for the estimation of the completeness magnitude.
"""

import warnings

import numpy as np

from seismostats.analysis.bvalue import estimate_b
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision, get_fmd
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


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
        max_considered_mag = np.max([np.max(sample), np.max(simulated_all)])

        x_bins = bin_to_precision(
            np.arange(mc, max_considered_mag + 3
                      / 2 * delta_m, delta_m), delta_m
        )
        x_bins -= delta_m / 2
        x = bin_to_precision((x_bins[1:] + x_bins[:-1]) / 2, delta_m)
        _, y_th = cdf_discrete_GR(x, mc=mc, delta_m=delta_m, beta=beta)

        for ii in range(n):
            simulated = simulated_all[n_sample * ii: n_sample * (ii + 1)]
            y_hist, _ = np.histogram(simulated, bins=x_bins)
            y_emp = np.cumsum(y_hist) / np.sum(y_hist)

            ks_d = np.max(np.abs(y_emp - y_th))
            ks_ds.append(ks_d)

    else:
        max_considered_mag = np.max(sample)
        x_bins = bin_to_precision(
            np.arange(mc, max_considered_mag + 3
                      / 2 * delta_m, delta_m), delta_m
        )

    y_hist, _ = np.histogram(sample, bins=x_bins)
    y_emp = np.cumsum(y_hist) / np.sum(y_hist)

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
    b_method: BValueEstimator = ClassicBValueEstimator,
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
        if beta is not None and verbose:
            print("Using given beta instead of estimating it.")

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
            estimator = b_method(mc=mc, delta_m=delta_m)
            mc_beta = estimator.estimate_beta(magnitudes=mc_sample)
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
        sample: np.ndarray,
        delta_m: float,
        stability_range: float = 0.5,
        mcs_test: np.ndarray | None = None,
        stop_when_passed: bool = True,
):
    """
    Estimates Mc using a test of stability.

    The stability of the b-value is tested by default on half a magnitude unit
    (in line with the 5x0.1 in the orginial paper). Users can change the range
    for the stability test by changing the stability_range.

    Source:
        Woessner, J, and Stefan W. "Assessing the quality of earthquake
        catalogues: Estimating the magnitude of completeness and its
        uncertainty." Bulletin of the Seismological Society of America 95.2
        (2005): 684-698.

    Args:
        sample:             Vector of magnitudes.
        delta_m:            Discretization of the magnitudes.
        stability_range:    Magnitude range to consider for the
            stability test. Default is 0.5 to consider half a magnitude unit,
            this is compatible with the original definition of Cao & Gao 2002.
        mcs_test:           Array of tested completeness magnitudes.
            If None, it will be generated automatically based on the sample and
            delta_m.
        stop_when_passed:   Whether to stop the stability test
            when a passing completeness magnitude (Mc) is found. Default is
            True.

    Returns:
        - best_mc:  Single best magnitude of completeness estimate.
        - best_b:   b-value associated with best_mc.
        - mcs_test: Array of tested completeness magnitudes.
        - bs:       Array of b-values associated to tested mcs
        - diff_bs:  Array of differences divided by std, associated with tested
            mcs. If a value is smaller than one, this means that the stability
            criterion is met.
    """
    # TODO: include a test if the sample is tested the correct way
    # instead of binning it here
    sample = bin_to_precision(sample, delta_x=delta_m)
    steps = len(np.arange(0, stability_range, delta_m))

    if mcs_test is None:
        mcs_test = np.arange(np.min(sample), np.max(sample), delta_m)
        mcs_test = bin_to_precision(mcs_test, delta_m)
        if len(mcs_test) <= steps:
            raise ValueError(
                "The range of magnitudes is smaller than the stability range."
            )
        mcs_test = mcs_test[:-steps + 1]
    else:
        mcs_test = mcs_test[mcs_test + stability_range <= np.max(sample)]
        if len(mcs_test) < 1:
            raise ValueError(
                "The range of magnitudes is smaller than the stability range."
            )

    bs = []
    diff_bs = []
    for ii, mc in enumerate(mcs_test):
        # TODO: here, one should be able to choose the method
        b, std = estimate_b(
            sample[sample >= mc - delta_m / 2], mc, delta_m,
            return_std=True)
        if len(sample[sample >= mc - delta_m / 2]) < 30:
            warnings.warn(
                "Number of events above tested Mc is less than 30. "
                "This might affect the stability test."
            )
        bs.append(b)

        mc_plus = np.arange(mc, mc + stability_range, delta_m)
        mc_plus = mc_plus[mc_plus <= np.max(sample)]
        b_ex = []
        for mc_p in mc_plus:
            # TODO: here, one should be able to choose the method
            b_p = estimate_b(sample[sample >= mc_p - delta_m / 2],
                             mc_p, delta_m)
            b_ex.append(b_p)
        b_avg = np.sum(b_ex) / steps
        diff_b = np.abs(b_avg - b) / std
        diff_bs.append(diff_b)
        if diff_b <= 1:
            value = True
            if diff_b == min(diff_bs):
                best_mc = mc
                best_b = bs[-1]
            if stop_when_passed:
                mcs_test = mcs_test[:ii + 1]
                break

    if value:
        return bin_to_precision(best_mc, delta_m), best_b, mcs_test, bs, diff_bs
    else:
        raise ValueError("No Mc passes the stability test")
