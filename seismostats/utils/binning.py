import decimal
import numpy as np


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
        tolerance: float = 1e-08,
        decimal_assumption: bool = True) -> float:
    """
    Finds out to which precision the given array is binned with delta_x.

    Args:
        x:          list of decimal numbers that are supposeddly binned
            (with bin-sizes delta_x)
        delta_x:    size of the bin
        tolerance:  tolerance for the comparison
        decimal_assumption:    if True (default), the function assumes that the
            array is binned to a power of ten. As an example, the function will
            return True for x =  [0, 0.2, 0.4], delta_x = 0.1, even though the
            bin size seems to be 0.2. If set to False, the function will be
            more strict, and only return True if delta_x = 0.2 for the example.

    Returns:
        result: True if the array is binned to the given precision, False
            otherwise.

    """
    if isinstance(x, list):
        x = np.array(x)
    x_binned = bin_to_precision(x, delta_x)

    # The first test can only be correct if the bins are <= delta_x
    test_1 = np.allclose(x_binned, x, atol=tolerance)
    if test_1:
        # second test checks if the bins are smaller than delta_x
        if decimal_assumption:
            factors = [10]
        else:
            factors = np.arange(2, 11)
        test_2 = True
        for factor in factors:
            x_binned = bin_to_precision(x, delta_x * factor)
            test_2 = test_2 and not np.allclose(x_binned, x, atol=tolerance)
    return test_1 and test_2


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
