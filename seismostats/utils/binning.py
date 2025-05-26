import decimal
import numpy as np


def _normal_round_to_int(x: float) -> int:
    """
    Rounds a float number ``x`` to the closest integer.

    Args:
        x:       Decimal number that needs to be rounded.

    Returns:
        x_round: Rounded value of the given number.

    Examples:
        >>> from seismostats.utils.binning import normal_round_to_int
        >>> _normal_round_to_int(2.5)
        3
        >>> _normal_round_to_int(2.4)
        2
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
        x:       Decimal number that needs to be rounded.
        n:       Number of decimals, optional.

    Returns:
        x_round: Value rounded to the given number of decimals.

    Examples:
        >>> from seismostats.utils.binning import normal_round
        >>> normal_round(2.123456, 2)
        2.12
        >>> normal_round(2.123456)
        2
    """

    power = 10**n
    return _normal_round_to_int(x * power) / power


def bin_to_precision(x: np.ndarray | list, delta_x: float) -> np.ndarray:
    """
    Rounds float numbers within the array ``x`` to a given precision. If
    precision is not given, an error is raised.

    Args:
        x:          List of decimal numbers that needs to be rounded.
        delta_x:    Size of the bin.

    Returns:
        x_round:   Value rounded to the given precision.

    Examples:
        >>> from seismostats.utils import bin_to_precision
        >>> bin_to_precision([1.234, 2.345, 3.456], 0.1)
        array([1.2, 2.3, 3.5])
        >>> bin_to_precision([1.234, 2.345, 3.456], 0.01)
        array([1.23, 2.35, 3.46])

    See also:
        :func:`~seismostats.utils.binning.normal_round`
    """
    if x is None:
        raise ValueError("x cannot be None")
    if delta_x == 0:
        raise ValueError("delta_x cannot be 0")

    if isinstance(x, list):
        x = np.array(x)

    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return np.round(_normal_round_to_int(x / delta_x) * delta_x, decimal_places)


def binning_test(
    x: np.ndarray | list,
    delta_x: float,
    tolerance: float = 1e-15,
    check_larger_binning: bool = True,
) -> float:
    """
    Finds out to which precision the given array is binned with ``delta_x``,
    within the given absolute tolerance.

    The function does have the implicit assumption of ``delta_x`` being a power
    of ten. As an example, what this means: the function will return True for
    ``x = [0, 0.2, 0.4]``, for ``delta_x = 0.2`` but also ``for delta_x = 0.1``.
    This is because the algorithm will check the next larger power of ten in
    order to determine if the array is binned to a larger ``delta_x``.

    If ``delta_x`` = 0, the function will test if the array is binned to a power
    of ten larger than the tolerance.

    Args:
        x:          List of decimal numbers that are supposeddly binned
                (with bin-sizes ``delta_x``).
        delta_x:    Size of the bin.
        tolerance:  Tolerance for the comparison. Default is 1e-15.
        check_larger_binning: If True (default), the function not only checks
                that the binning of the array is correct, but also makes sure
                that there is no other binning that is correct. For example,
                take the array [1.0, 3.0, 4.0]. If ``delta_x = 0.1``, the
                function will return False because a larger binning (1.0) is
                also correct. Here, it is important to note that only the next
                larger power of ten is checked. In case of
                ``check_larger_binning = False``, the function will return
                True for the example above, as the binning of 0.1 is correct,
                and the larger binning is not checked.

    Returns:
        result:     True if the array is binned to the given precision, False
                otherwise.

    Examples:
        >>> from seismostats.utils.binning import binning_test
        >>> binning_test([0.2,0.4,0.6], 0.2)
        True
        >>> binning_test([0.2,0.4,0.6], 0.1)
        True
        >>> binning_test([0.2,0.4,0.6], 0.05)
        False
        >>> binning_test([0.2,0.4,0.6], 0.05, check_larger_binning=False)
        True

    See also:
        :func:`~seismostats.utils.binning.bin_to_precision`
    """
    if len(x) == 0:
        # error if the array is empty
        raise ValueError("The given array has no entry")
    x = np.array(x)

    # filter out NaN values
    x = x[~np.isnan(x)]

    # shift the array to the smallest value (to avoid that the bin-center has
    # an effect on the test)
    x = x - np.min(x)

    if delta_x == 0 and check_larger_binning is True:
        range = np.max(x) - np.min(x)
        power = np.arange(
            np.floor(np.log10(tolerance)) + 1, np.ceil(np.log10(range)), 1
        )
        delta_x_test = 10**power
        test = True
        tolerance = 10 ** (np.floor(np.log10(tolerance)) - 1)
        for delta_x_loop in delta_x_test:
            if binning_test(x, delta_x_loop, tolerance):
                return False

    else:
        if delta_x < tolerance:
            return True
        x_binned = bin_to_precision(x, delta_x)

        # This test checks if the bins are equal or larger than delta_x.
        test_1 = np.allclose(x_binned, x, atol=tolerance, rtol=1e-16)

        # If the test_1 is True, we check if a larger binning is correct.
        if test_1 and check_larger_binning is True:
            power = np.floor(np.log10(delta_x)) + 1
            x_binned = bin_to_precision(x, 10**power)
            test_2 = not np.allclose(x_binned, x, atol=tolerance, rtol=1e-16)
            test = test_1 and test_2
        else:
            test = test_1
    return test


def get_fmd(
    magnitudes: np.ndarray, fmd_bin: float, bin_position: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates event counts per magnitude bin. Note that the returned bins
    array contains the center point of each bin unless
    ``bin_position = 'left'``.

    Args:
        mags:           Array of magnitudes.
        fmd_bin:        Bin size for the FMD. This can be independent of
                    the discretization of the magnitudes. The optimal value
                    would be as small as possible while at the same time
                    ensuring that there are enough magnitudes in each bin.
        bin_position:   Position of the bin, options are  'center' and 'left'.
                    Accordingly, left edges of bins or center points are
                    returned.
    Returns:
        bins:           Array of bin centers (left to right).
        counts:         Counts for each bin.
        mags:           Array of magnitudes binned to ``fmd_bin``.

    Examples:
        >>> from seismostats.utils import get_fmd

        >>> magnitudes = [0.9, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
        >>> fmd_bin = 1.0
        >>> bin_position = "center"
        >>> bins, counts, mags = get_fmd(magnitudes, fmd_bin, bin_position)
        >>> bins
        array([1., 2.])
        >>> counts
        array([4, 3])
        >>> mags
        array([1., 1., 1., 1., 2., 2., 2.])

    See also:
        :func:`~seismostats.utils.binning.get_cum_fmd`
    """

    if fmd_bin <= 0:
        raise ValueError("Bin size (fmd_bin) must be a positive number.")

    magnitudes = bin_to_precision(magnitudes, fmd_bin)
    # use histogram to get the counts
    x_bins = bin_to_precision(
        np.arange(
            np.min(magnitudes), np.max(magnitudes) + 3 / 2 * fmd_bin, fmd_bin
        ),
        fmd_bin,
    )
    bins = x_bins[:-1].copy()
    x_bins -= fmd_bin / 2
    counts, _ = np.histogram(magnitudes, x_bins)

    assert (
        bin_position == "left" or bin_position == "center"
    ), "bin_position needs to be 'left'  of 'center'"
    if bin_position == "left":
        bins = bins - fmd_bin / 2

    return bins, counts, magnitudes


def get_cum_fmd(
    mags: np.ndarray, delta_m: float, bin_position: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates cumulative event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless ``bin_position = 'left'`` is used.

    Args:
        mags:           Array of magnitudes.
        delta_m:        Discretization of the magnitudes. It is possible to
                    provide a value that is larger than the actual
                    discretization of the magnitudes. In this case, the
                    magnitudes will be binned to the given ``delta_m``. This
                    might be useful for visualization purposes.
        bin_position:   Position of the bin, options are  'center' and 'left'.
                    Accordingly, left edges of bins or center points are
                    returned.

    Returns:
        bins:           Array of bin centers (left to right).
        c_counts:       Cumulative counts for each bin.
        mags:           Array of magnitudes binned to ``delta_m``.

    Examples:
        >>> from seismostats.utils import get_cum_fmd

        >>> magnitudes = [0.9, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
        >>> delta_m = 1.0
        >>> bin_position = "center"
        >>> bins, counts, mags = get_cum_fmd(magnitudes, delta_m, bin_position)
        >>> bins
        array([1., 2.])
        >>> counts
        array([7, 3])
        >>> mags
        array([1., 1., 1., 2., 2., 2., 2.])

    See also:
        :func:`~seismostats.utils.binning.get_fmd`
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
