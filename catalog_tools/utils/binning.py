import numpy as np
import decimal
from typing import Union


def normal_round_to_int(x: float) -> int:
    """
    Rounds a float number x to the closest integer.

    Args:
        x: decimal number that needs to be rounded

    Returns:
        Rounded value of the given number.
    """

    sign = np.sign(x)
    y = abs(x)
    y = np.floor(y + 0.5)

    return sign * y


def normal_round(x: float, n: int = 0) -> float:
    """
    Rounds a float number x to n number of decimals. If the number
    of decimals is not given, we round to an integer.

    Args:
        x: decimal number that needs to be rounded
        n: number of decimals, optional

    Returns:
        Value rounded to the given number of decimals.
    """

    power = 10**n
    return normal_round_to_int(x * power) / power


def bin_to_precision(x: Union[np.ndarray, list], delta_x: float = 0.1
                     ) -> np.ndarray:
    """
    Rounds a float number x to a given precision. If precision not given,
    assumes 0.1 bin size

    Args:
        x: decimal number that needs to be rounded
        delta_x: size of the bin, optional

    Returns:
        Value rounded to the given precision.
    """
    if type(x) == list:
        x = np.array(x)
    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return np.round(normal_round_to_int(x / delta_x) * delta_x, decimal_places)


def get_fmd(
        mags: np.ndarray,
        delta_m: float,
        left: bool = False
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Calculates event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless left is True.

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes
        left    : When True, left edges of bins are returned. When false,
                center points are returned.

    Returns:
        bins    : array of bin centers (left to right)
        counts  : counts for each bin ("")
        mags    : array of magnitudes binned to delta_m
    """
    mags = bin_to_precision(mags, delta_m)
    mags_i = bin_to_precision(mags / delta_m - np.min(mags / delta_m), 1)
    mags_i = mags_i.astype(int)
    counts = np.bincount(mags_i)
    bins = bin_to_precision(np.arange((np.min(mags)) * 10000,
                                      (np.max(mags) + delta_m / 2) * 10000,
                                      delta_m * 10000) / 10000, delta_m)

    if left:
        bins = bins - delta_m / 2

    return bins, counts, mags


def get_cum_fmd(
        mags: np.ndarray,
        delta_m: float,
        left: bool = False
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Calculates cumulative event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless left is True.

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes
        left    : When True, left edges of bins are returned. When false,
                center points are returned.

    Returns:
        bins    : array of bin centers (left to right)
        c_counts: cumulative counts for each bin ("")
        mags    : array of magnitudes binned to delta_m
    """
    bins, counts, mags = get_fmd(mags, delta_m, left)

    c_counts = np.cumsum(counts[::-1])
    c_counts = c_counts[::-1]

    return bins, c_counts, mags
