import numpy as np
from scipy import stats

from seismostats.utils.binning import bin_to_precision


def simulate_magnitudes(
    n: int, beta: float, mc: float, mag_max: float | None = None
) -> np.ndarray:
    """
    Generates a vector of ``n`` elements drawn from an exponential distribution
    :math:`f = e^{-beta*M}`.

    Args:
        n:          Number of sample magnitudes.
        beta:       Scale factor of the exponential distribution.
        mc:         Cut-off magnitude.
        mag_max:    Maximum magnitude. If it is not None, the exponential
                distribution is truncated at ``mag_max``.

    Returns:
        mags:       Vector of length ``n`` of magnitudes drawn from an
                exponential distribution.

    Examples:
        >>> from seismostats.utils import simulate_magnitudes
        >>> simulate_magnitudes(4, 1, 0, 5)
        array([1.39701219, 0.09509761, 2.68367219, 0.73664695]) #random
        >>> simulate_magnitudes(4, 1, 1)
        array([1.3249975 , 1.63120196, 3.56443043, 1.15384524]) #random

    See also:
        :func:`~seismostats.utils.simulate_distributions.simulate_magnitudes_binned`

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
    """
    Simulate magnitudes and bin them to a given precision ``delta_m``.
    Input ``b`` can be specified to be 'beta' or the 'b-value',
    depending on the ``b_parameter`` input.

    Args:
        n:              Number of magnitudes to simulate.
        b:              b-value or beta of the distribution from which
                    magnitudes are simulated. If ``b`` is np.ndarray, it must
                    have the length ``n``. Then each magnitude is simulated
                    from the corresponding b-value.
        mc:             Magnitude of completeness.
        delta_m:        Magnitude bin width.
        mag_max:        Maximum magnitude. If it is not None, the exponential
                distribution is truncated at ``mag_max``.
        b_parameter:    'b_value' or 'beta'

    Returns:
        mags:           Array of magnitudes.

    Examples:
        >>> from seismostats.utils import simulate_magnitudes_binned
        >>> simulate_magnitudes_binned(5, 1, 0, 1, 5)
        array([1., 0., 1., 1., 0.])
        >>> simulate_magnitudes_binned(5, 1, 1, 0.1)
        array([1.1., 1., 1.6, 1.2, 1.3])

    See also:
        :func:`~seismostats.utils.simulate_distributions.simulate_magnitudes`
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
