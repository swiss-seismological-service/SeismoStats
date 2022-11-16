import numpy as np


def simulate_magnitudes(n: int, beta: float, mc: float) -> np.ndarray:
    """ Generates a vector of n elements drawn from an exponential distribution
    exp(-beta*M)

    Args:
        n:      number of sample magnitudes
        beta:   scale factor of the exponential distribution
        mc:     completion magnitude

    Returns:
        mags:   vector of length n of magnitudes drawn from an exponential
        distribution
    """

    mags = np.random.exponential(1 / beta, n) + mc

    return mags
