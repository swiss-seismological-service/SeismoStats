"""This module contains functions for the estimation of beta and the b-value.
"""
import numpy as np
from typing import Optional


def estimate_beta_tinti(magnitudes: np.ndarray,
                        mc: float,
                        delta_m: float = 0,
                        weights: Optional[list] = None,
                        gutenberg: bool = False,
                        error: bool = False
                        ) -> (float, float):
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
        gutenberg:  if True the b-value of the Gutenberg-Richter law is
                    returned, otherwise the beta value of the exponential
                    distribution [p(M) = exp(-beta*(M-mc))] is returned
        error:      if True the error of beta/b-value (see above) is returned

    Returns:
        b:          maximum likelihood beta or b-value, depending on value of
                    input variable 'gutenberg'. Note that the difference
                    is just a factor [b_value = beta * log10(e)]
        std_b:      Shi and Bolt estimate of the beta/b-value estimate
    """

    if delta_m > 0:
        p = 1 + delta_m / np.average(magnitudes - mc, weights=weights)
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(magnitudes - mc, weights=weights)

    if gutenberg is True:
        factor = 1 / np.log(10)
    else:
        factor = 1

    if error is True:
        std_b = shi_bolt_confidence(magnitudes, beta=beta) * factor
        b = beta * factor
        return b, std_b
    else:
        b = beta * factor
        return b


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
    mag_diffs = np.diff(magnitudes)
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
    mag_diffs = mag_diffs[mag_diffs > 0]
    beta = estimate_beta_tinti(mag_diffs, mc=delta_m, delta_m=delta_m)
    return beta


def shi_bolt_confidence(
        magnitudes: np.ndarray,
        b_value: Optional[float] = None,
        beta: Optional[float] = None
) -> float:
    """ calculates the confidence limit of the b_value or beta (depending on
        which parameter is given) according to shi and bolt 1982

    Source:
        Shi and Bolt, BSSA, Vol. 72, No. 5, pp. 1677-1687, October 1982

    Args:
        magnitudes: numpy array of magnitudes
        b_value:    b-value of the magnitudes
        beta:       beta value (difference to b-value is factor of np.log(10)).
                    -> provide either b_value or beta, not both

    Returns:
        sig_b:  confidence limit of the b-value/beta value (depending on input)
    """
    # standard deviation in Shi and Bolt is calculated with 1/(N*(N-1)), which
    # is by a factor of sqrt(N) different to the std(x, ddof=1) estimator
    assert b_value is not None or beta is not None,\
        'please specify b-value or beta'
    assert b_value is None or beta is None, \
        'please only specify either b-value or beta'

    if b_value is not None:
        std_m = np.std(magnitudes, ddof=1) / np.sqrt(len(magnitudes))
        std_b = np.log(10) * b_value ** 2 * std_m
    else:
        std_m = np.std(magnitudes, ddof=1) / np.sqrt(len(magnitudes))
        std_b = beta ** 2 * std_m

    return std_b
