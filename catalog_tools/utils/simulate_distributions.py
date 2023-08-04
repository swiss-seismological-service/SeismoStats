import numpy as np
from scipy import stats
from typing import Optional, Tuple, Union


def simulate_magnitudes(n: int, beta: float, mc: float,
                        mag_max: Union[float, None] = None) -> np.ndarray:
    """ Generates a vector of n elements drawn from an exponential distribution
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
