"""This module contains functions
for the estimation of the completeness magnitude.
"""

import warnings

import numpy as np

from seismostats.analysis.bvalue import estimate_b
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.bvalue.utils import b_value_to_beta
from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision, binning_test, get_fmd
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def cdf_discrete_exp(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cumulative distribution function (CDF) for a discrete
    exponential distribution at the points of the sample.

    Args:
        sample:     Array of magnitudes.
        mc:         Completeness magnitude.
        delta_m:    Bin size of discretized magnitudes.
        beta:       Rate parameter of the exponential distribution.

    Returns:
        x: Unique x-values of the sample.
        y: Corresponding y-values of the CDF of the GR distribution.
    """

    x = np.sort(sample)
    x = np.unique(x)
    y = 1 - np.exp(-beta * (x + delta_m - mc))
    return x, y


def ks_test_gr(
    sample: np.ndarray,
    mc: float,
    delta_m: float,
    b_value: float,
    n: int = 10000,
    ks_ds: list | None = None,
) -> tuple[float, float, list[float]]:
    """
    Performs the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution for a given magnitude sample and mc and b-value. When the
    p-value is below a certain threshold (e.g., 0.1), the null hypothesis that
    the sample is drawn from a Gutenberg-Richter distribution with the given
    parameters can be rejected.

    Args:
        sample:     Array of magnitudes.
        mc:         Completeness magnitude.
        delta_m:    Bin size of discretized magnitudes.
        b_value:    b-value of the Gutenberg-Richter law.
        n:          Number of times the KS distance is calculated from
                synthetic samples with the given parameters, used for
                estimating the p-value.
        ks_ds:      KS distances from synthetic data with the given
                paramters. If None, they will be estimated here (then, n is
                not needed).

    Returns:
        p_val:      p-value.
        ks_d_obs:   KS distance of the sample.
        ks_ds:      Array of KS distances.
    """
    if get_option("warnings") is True:
        if np.min(sample) < mc - delta_m / 2:
            warnings.warn("Sample contains values below mc.")

        if len(sample) == 0:
            warnings.warn("No sample given.")
            return 0, 1, []

        if len(np.unique(sample)) == 1:
            warnings.warn("Sample contains only one value.")
            return 0, 1, []

    beta = b_value_to_beta(b_value)

    if ks_ds is None:
        ks_ds = []

        n_sample = len(sample)
        simulated_all = simulate_magnitudes_binned(
            n * n_sample, b_value, mc, delta_m, b_parameter="b_value"
        )
        max_considered_mag = np.max([np.max(sample), np.max(simulated_all)])

        x_bins = bin_to_precision(
            np.arange(mc, max_considered_mag + 3
                      / 2 * delta_m, delta_m), delta_m
        )
        x = x_bins[:-1].copy()
        x_bins -= delta_m / 2
        _, y_th = cdf_discrete_exp(
            x, mc=mc, delta_m=delta_m, beta=beta)

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
        x = x_bins[:-1].copy()
        x_bins -= delta_m / 2
        _, y_th = cdf_discrete_exp(x, mc=mc, delta_m=delta_m, beta=beta)

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
    b_value: float | None = None,
    b_method: BValueEstimator = ClassicBValueEstimator,
    n: int = 10000,
    ks_ds_list: list[list] | None = None,
    **kwargs,
) -> tuple[float | None, float | None, list[float],
           list[float], list[float], np.ndarray]:
    """
    Returns the smallest magnitude in a given list of completeness magnitudes
    for which the KS test is passed, i.e., where the null hypothesis that the
    sample is drawn from a Gutenberg-Richter law with that mc cannot be
    rejected.

    Source:
        - Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law
          distributions in empirical data. SIAM review, 51(4), pp.661-703.
        - Mizrahi, L., Nandan, S. and Wiemer, S., 2021. The effect of
          declustering on the size distribution of mainshocks. Seismological
          Society of America, 92(4), pp.2333-2342.

    Args:
        sample:             Array of magnitudes to test.
        delta_m:            Bin size of discretized magnitudes. Sample has to be
                        rounded to bins beforehand).
        mcs_test:           Array of tested completeness magnitudes. If `None`,
                        it will be generated automatically based on `sample`
                        and `delta_m`.
        p_pass:             p-value threshold for the KS test. Below this value,
                        the null hypothesis that the sample is drawn from a
                        Gutenberg-Richter distribution with the given mc is
                        rejected.
        stop_when_passed:   Boolean that indicates whether to stop calculations
                        when first mc passes the test.
        verbose:            Boolean that indicates whether to print verbose
                        output.
        b_value:            If `b_value` is 'known', only estimate `mc` assuming
                        the given `b_value`.
        b_method:           b-value estimator to use if b-value needs to be
                        calculated from data
        n:                  Number of number of times the KS distance is
                        calculated for estimating the p-value.
        ks_ds_list:         KS distances from synthetic data with the given
                        parameters. If `None`, they will be estimated here.
        **kwargs:           Additional parameters to be passed to the b-value
                        estimator.

    Returns:
        best_mc:        `mc` for which the p-value is lowest.
        best_b_value:   `b_value` corresponding to the best `mc`.
        mcs_test:       Tested completeness magnitudes.
        b_values:       Tested b-values.
        ks_ds:          KS distances.
        ps:             Corresponding p-values.
    """

    if mcs_test is None:
        mcs_test = bin_to_precision(
            np.arange(np.min(sample), np.max(sample), delta_m), delta_m
        )
    else:
        # sort mcs
        mcs_test = np.sort(np.unique(mcs_test))

    if get_option("warnings") is True:
        # check if binning is correct
        if not binning_test(sample, delta_m, check_larger_binning=False):
            warnings.warn(
                "Magnitudes are not binned correctly. "
                "Test might fail because of this."
            )
        if not binning_test(mcs_test, delta_m, check_larger_binning=False):
            warnings.warn(
                "Mcs to test are not binned correctly. "
                "Test might fail because of this."
            )

        # check if b-value is given (then b_method is not needed)
        if b_value is not None and verbose:
            print("Using given b-value instead of estimating it.")

    mcs_tested = []
    ks_ds = []
    ps = []
    b_values = []

    for ii, mc in enumerate(mcs_test):

        if verbose:
            print("\ntesting mc", mc)

        mc_sample = sample[sample >= mc - delta_m / 2]

        # if no b_value is given, estimate b_value
        if b_value is None:
            estimator = b_method()
            estimator.calculate(
                mc_sample, mc=mc, delta_m=delta_m, **kwargs)
            mc_b_value = estimator.b_value
        else:
            mc_b_value = b_value

        if ks_ds_list is None:
            p, ks_d, _ = ks_test_gr(
                mc_sample, mc=mc, delta_m=delta_m, b_value=mc_b_value, n=n
            )
        else:
            p, ks_d, _ = ks_test_gr(
                mc_sample,
                mc=mc,
                delta_m=delta_m,
                b_value=mc_b_value,
                n=n,
                ks_ds=ks_ds_list[ii],
            )

        mcs_tested.append(mc)
        ks_ds.append(ks_d)
        ps.append(p)
        b_values.append(mc_b_value)

        if verbose:
            print("..p-value: ", p)

        if p >= p_pass and stop_when_passed:
            break

    ps = np.array(ps)

    if np.any(ps >= p_pass):
        best_mc = mcs_tested[np.argmax(ps >= p_pass)]
        best_b_value = b_values[np.argmax(ps >= p_pass)]

        if verbose:
            print(
                "\n\nFirst mc to pass the test:",
                best_mc,
                "\nwith a b-value of:",
                best_b_value,
            )
    else:
        best_mc = None
        best_b_value = None

        if verbose:
            print("None of the mcs passed the test.")

    return best_mc, best_b_value, mcs_tested, b_values, ks_ds, ps


def mc_max_curvature(
    sample: np.ndarray,
    delta_m: float,
    correction_factor: float = 0.2,
) -> float:
    """
    Returns the completeness magnitude (mc) estimate using the maximum
    curvature method.

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
        sample:             Array of magnitudes to test.
        delta_m:            Bin size of discretized magnitudes. Sample has to be
                        rounded to bins beforehand).
        correction_factor:  Correction factor for the maximum curvature
                method (default value after Woessner & Wiemer 2005).

    Returns:
        mc:                 Estimated completeness magnitude.
    """
    bins, count, _ = get_fmd(
        magnitudes=sample, delta_m=delta_m, bin_position="center"
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
    Estimates the completeness magnitude (mc) using b-value stability.

    The stability of the b-value is tested by default on half a magnitude unit
    (in line with the 5x0.1 in the orginial paper). Users can change the range
    for the stability test by changing the stability_range.

    Source:
        Woessner, J, and Stefan W. "Assessing the quality of earthquake
        catalogues: Estimating the magnitude of completeness and its
        uncertainty." Bulletin of the Seismological Society of America 95.2
        (2005): 684-698.

    Args:
        sample:             Array of magnitudes.
        delta_m:            Bin size of discretized magnitudes. Sample has to be
                        rounded to bins beforehand.
        stability_range:    Magnitude range to consider for the stability test.
                        Default compatible with the original definition of
                        Cao & Gao 2002.
        mcs_test:           Array of tested completeness magnitudes. If None,
                        it will be generated automatically based on the sample
                        and delta_m.
        stop_when_passed:   Boolean that indicates whether to stop computation
                        when a completeness magnitude (mc) has passed the test.

    Returns:
        best_mc:    Best magnitude of completeness estimate.
        best_b:     b-value associated with best_mc.
        mcs_test:   Array of tested completeness magnitudes.
        bs:         Array of b-values associated to tested mcs.
        diff_bs:    Array of differences divided by std, associated with tested
                mcs. If a value is smaller than one, this means that the
                stability criterion is met.
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
        raise ValueError("No Mc passes the stability test.")
