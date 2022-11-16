import numpy as np


def estimate_beta_tinti(magnitudes: np.ndarray, mc: float, delta_m: float = 0,
                        weights: list = None) -> float:
    """ returns the maximum likelihood beta
    Source:
        Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
            America, 77(6), 2125-2134.)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no
                    magnitudes below mc present)
        mc:         completeness magnitude
        delta_m:    discretization of magnitudes. default is no discretization
        weights: weights of each magnitude can be specified here

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """

    if delta_m > 0:
        p = (1 + (delta_m / (np.average(magnitudes - mc, weights=weights))))
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average((magnitudes - mc), weights=weights)

    return beta


def estimate_beta_utsu(magnitudes: np.ndarray, mc: float, delta_m: float = 0) \
        -> float:
    """ returns the maximum likelihood beta
    Source:
        Utsu 1965 (Geophysical bulletin of the Hokkaido University, vol 13, pp
        99-103)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no
                    magnitudes below mc present)
        mc:         completeness magnitude
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """

    beta = 1 / (np.mean(magnitudes) - mc - delta_m / 2)

    return beta


def estimate_beta_elst(magnitudes: np.ndarray) -> float:
    """ returns the b-value estimation using the positive differences of the
    Magnitudes

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """
    temp_mags1 = np.append([0], magnitudes)
    temp_mags2 = np.append(magnitudes, [0])
    mag_diffs = temp_mags1 - temp_mags2
    mag_diffs = mag_diffs[1:-1]

    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs < 0])

    b_value = estimate_beta_utsu(mag_diffs, mc=0.0, delta_m=0.0)

    return b_value
