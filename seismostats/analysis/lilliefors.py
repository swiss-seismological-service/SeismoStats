import numpy as np
import statsmodels.api as sm


def ks_test_gr_lilliefors(
    magnitudes: np.ndarray,
    mc: float,
) -> float:
    """
    Performs the Kolmogorov-Smirnov (KS) test for the Gutenberg-Richter
    distribution for a given magnitude sample and mc and b-value, based on
    the Lilliefors test. The magnitudes are assumed to be continuous. In order
    to use the test for discrete magnitudes, the test should be performed
    multiple times with different dithered magnitudes.
    Source:
        - Lilliefors, Hubert W. "On the Kolmogorov-Smirnov test for the
        exponential distribution with mean unknown." Journal of the American
        Statistical Association 64.325 (1969): 387-389.
    Args:
        magnitudes: Array of magnitudes, should be continuous.
        mc:         Completeness magnitude. Make sure that no magnitudes below
                    mc are present in the array.
    Returns:
        p_val:      p-value.
    """
    if min(magnitudes) < mc:
        raise ValueError(
            'The minimum magnitude in the array is smaller than mc. '
        )

    mags_shifted = magnitudes - mc
    out = sm.stats.diagnostic.lilliefors(
        mags_shifted, dist='exp', pvalmethod='table')
    p_val = out[1]

    return p_val
