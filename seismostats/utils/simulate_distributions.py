import numpy as np
from scipy import stats
from seismostats.utils.binning import bin_to_precision


def simulate_magnitudes(
    n: int, beta: float, mc: float, mag_max: float | None = None
) -> np.ndarray:
    """Generates a vector of n elements drawn from an exponential distribution
    exp(-beta*M)

    Args:
        n:      number of sample magnitudes
        beta:   scale factor of the exponential distribution
        mc:     cut-off magnitude
        mag_max: maximum magnitude. If it is not None, the exponential
                 distribution is truncated at mag_max.

    Returns:
        mags:   vector of length n of magnitudes drawn from an exponential
        distribution
    """
    if mag_max:
        quantile1 = stats.expon.cdf(mc, loc=0, scale=1 / beta)
        quantile2 = stats.expon.cdf(mag_max, loc=0, scale=1 / beta)
        mags = stats.expon.ppf(
            np.random.uniform(quantile1, quantile2, size=n),
            loc=0,
            scale=1 / beta,
        )
    else:
        mags = np.random.exponential(1 / beta, n) + mc

    return mags


def simulate_magnitudes_binned(
    n: int,
    b: float | np.ndarray,
    mc: float,
    delta_m: float,
    mag_max: float = None,
    b_parameter: str = "b_value",
) -> np.ndarray:
    """simulate magnitudes and bin them to a given precision. input 'b' can be
    specified to be beta or the b-value, depending on the 'b_parameter' input

    Args:
        n:              number of magnitudes to simulate
        b:              b-value or beta of the distribution from which
                magnitudes are simulated. If b is np.ndarray, it must have the
                length n. Then each magnitude is simulated from the
                corresponding b-value
        mc:             completeness magnitude
        delta_m:        magnitude bin width
        mag_max:        maximum magnitude
        b_parameter:    'b_value' or 'beta'

    Returns:
        mags:   array of magnitudes
    """
    if b_parameter == "b_value":
        beta = b * np.log(10)
    elif b_parameter == "beta":
        beta = b
    else:
        raise ValueError("b_parameter must be 'b_value' or 'beta'")

    mags = simulate_magnitudes(n, beta, mc - delta_m / 2, mag_max)
    if delta_m > 0:
        mags = bin_to_precision(mags, delta_m)
    return mags
