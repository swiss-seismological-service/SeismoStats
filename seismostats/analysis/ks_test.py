import warnings
import numpy as np

from seismostats.analysis.bvalue.utils import b_value_to_beta
from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


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

        # max considered magnitude: less than 1e-3 probability of
        # exceedance within the  samples
        max_considered_mag = 1 / b_value * np.log10(n_sample * n * 1e3) + mc

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
