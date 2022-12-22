"""This module contains functions for the estimation of beta and the b-value.
"""
import numpy as np
from typing import Optional


def estimate_beta_tinti(magnitudes: np.ndarray, mc: float, delta_m: float = 0,
                        weights: Optional[list] = None) -> float:
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
        weights:    weights of each magnitude can be specified here

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """

    if delta_m > 0:
        p = 1 + delta_m / np.average(magnitudes - mc, weights=weights)
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(magnitudes - mc, weights=weights)

    return beta


def estimate_beta_utsu(magnitudes: np.ndarray, mc: float, delta_m: float = 0
                       ) -> float:
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
    beta = 1 / np.mean(magnitudes - mc + delta_m / 2)

    return beta


def differences(magnitudes: np.ndarray) -> np.ndarray:
    """returns all the differences between the magnitudes.

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)

    Returns: array of all differences of the elements of the input
    """
    mag_diffs = np.array([])
    for ii, mag in enumerate(magnitudes):
        loop_mag = np.delete(magnitudes, [ii], axis=0)
        mag_diffs = np.append(mag_diffs, loop_mag - mag)
    return mag_diffs


def differences_successive(magnitudes: np.ndarray) -> np.ndarray:
    """returns the differences of successive magnitudes.

    Args:
        magnitudes: vector of magnitudes, sorted in time (first
                    entry is the earliest earthquake)

    Returns: array of successive differences of the input
    """
    temp_mags1 = np.append([0], magnitudes)
    temp_mags2 = np.append(magnitudes, [0])
    mag_diffs = temp_mags2 - temp_mags1
    mag_diffs = mag_diffs[1:-1]

    return mag_diffs


def estimate_beta_elst(magnitudes: np.ndarray, delta_m: float = 0
                       ) -> [float, float]:
    """ returns the b-value estimation using the positive differences of the
    Magnitudes

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """
    mag_diffs = differences_successive(magnitudes)
    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs > 0])
    beta = estimate_beta_tinti(mag_diffs, mc=delta_m, delta_m=delta_m)

    return beta


def estimate_beta_laplace(
        magnitudes: np.ndarray,
        delta_m: float = 0
) -> float:
    """ returns the b-value estimation using the all the  differences of the
    Magnitudes (this has a little less variance than the estimate_beta_elst
    method)

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """
    mag_diffs = differences(magnitudes)
    mag_diffs = abs(mag_diffs)

    if delta_m > 0:
        a = np.average(mag_diffs) / delta_m
        p = (1 + np.sqrt(a ** 2 + 1)) / a
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(mag_diffs)

    return beta
