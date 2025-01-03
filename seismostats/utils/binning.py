import decimal
import numpy as np
import warnings

from seismostats.utils._config import get_option


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
    Rounds a float number ``x`` to n number of decimals. If the number
    of decimals is not given, we round to an integer.

    Args:
        x: decimal number that needs to be rounded
        n: number of decimals, optional

    Returns:
        Value rounded to the given number of decimals.
    """

    power = 10**n
    return normal_round_to_int(x * power) / power


def bin_to_precision(x: np.ndarray | list, delta_x: float) -> np.ndarray:
    """
    Rounds float numbers within the array ``x`` to a given precision. If
    precision not given, throws error.

    Args:
        x: list of decimal numbers that needs to be rounded
        delta_x: size of the bin

    Returns:
        Value rounded to the given precision.
    """
    if x is None:
        raise ValueError("x cannot be None")
    if delta_x == 0:
        raise ValueError("delta_x cannot be 0")

    if isinstance(x, list):
        x = np.array(x)

    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return np.round(normal_round_to_int(x / delta_x) * delta_x, decimal_places)


def binning_test(
        x: np.ndarray | list,
        delta_x: float,
        tolerance: float = 1e-08) -> float:
    """
    Finds out to which precision the given array is binned with delta_x,
      within the given absolute tolerance.

    The function does have the implicit assumption of delta_x being a power
    of ten. As an example, what this means: the function will return True for
    x =  [0, 0.2, 0.4], for delta_x = 0.2 but also for delta_x = 0.1. This is
    because the algorithm will check the next larger power of ten in order
    to determine if the array is binned to a larger delta_x.

    If delta_x == 0, the function will test if the array is binned to a power
    of ten larger than the tolerance.

    Args:
        x:          list of decimal numbers that are supposeddly binned
            (with bin-sizes delta_x)
        delta_x:    size of the bin
        tolerance:  tolerance for the comparison

    Returns:
        result: True if the array is binned to the given precision, False
            otherwise.

    """
    if delta_x == 0:
        range = np.max(x) - np.min(x)
        power = np.arange(np.floor(np.log10(tolerance)) + 1,
                          np.ceil(np.log10(range)), 1)
        delta_x_test = 10**power
        test = True
        tolerance = 10**(np.floor(np.log10(tolerance)) - 1)
        for delta_x_loop in delta_x_test:
            print(binning_test(x, delta_x_loop, tolerance))
            if binning_test(x, delta_x_loop, tolerance):
                return False

    else:
        x = np.asarray(x)
        x_binned = bin_to_precision(x, delta_x)

        # The first test can only be correct if the bins are <= delta_x
        if delta_x <= tolerance:
            if get_option("warnings") is True:
                warnings.warn(
                    "tolerance is smaller than binning, returning True by"
                    "default")
            return True
        test_1 = np.allclose(x_binned, x, atol=tolerance, rtol=1e-16)
        if test_1:
            # second test checks if the bins are smaller than delta_x
            # For this, we check the next larger power of ten
            power = np.floor(np.log10(delta_x)) + 1
            x_binned = bin_to_precision(x, 10**power)
            test_2 = not np.allclose(x_binned, x, atol=tolerance, rtol=1e-16)
        test = test_1 and test_2
    return test


def get_fmd(
    mags: np.ndarray, delta_m: float, bin_position: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates event counts per magnitude bin. Note that the returned bins
    array contains the center point of each bin unless
    ``bin_position = 'left'``.

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes
        bin_position    : position of the bin, options are  'center' and 'left'.
                        accordingly, left edges of bins or center points are
                        returned.
    Returns:
        bins    : array of bin centers (left to right)
        counts  : counts for each bin
        mags    : array of magnitudes binned to ``delta_m``
    """
    mags = bin_to_precision(mags, delta_m)
    mags_i = bin_to_precision(mags / delta_m - np.min(mags / delta_m), 1)
    mags_i = mags_i.astype(int)
    counts = np.bincount(mags_i)

    bins = bin_to_precision(
        np.arange(
            (np.min(mags)) * 100000,
            (np.max(mags) + delta_m / 2) * 100000,
            delta_m * 100000,
        )
        / 100000,
        delta_m,
    )

    assert (
        bin_position == "left" or bin_position == "center"
    ), "bin_position needs to be 'left'  of 'center'"
    if bin_position == "left":
        bins = bins - delta_m / 2

    return bins, counts, mags


def get_cum_fmd(
    mags: np.ndarray, delta_m: float, bin_position: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates cumulative event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless ``bin_position = 'left'``.

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes
        bin_position    : position of the bin, options are  'center' and 'left'.
                        accordingly, left edges of bins or center points are
                        returned.

    Returns:
        bins    : array of bin centers (left to right)
        c_counts: cumulative counts for each bin
        mags    : array of magnitudes binned to ``delta_m``
    """

    if delta_m == 0:
        mags_unique, counts = np.unique(mags, return_counts=True)
        idx = np.argsort(mags_unique)
        bins = mags_unique
        counts = counts[idx]
    else:
        bins, counts, mags = get_fmd(mags, delta_m, bin_position=bin_position)
    c_counts = np.cumsum(counts[::-1])
    c_counts = c_counts[::-1]

    return bins, c_counts, mags
