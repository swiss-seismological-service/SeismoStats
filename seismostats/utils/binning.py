import decimal
import math
import numpy as np

EPSILON = 1e-12


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

    return np.floor(x + 0.5 + EPSILON)


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
    if delta_x < EPSILON:
        raise ValueError(f"delta_x cannot be 0 or lower than {EPSILON}")

    if isinstance(x, list):
        x = np.array(x)

    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return normal_round(_normal_round_to_int(x / delta_x) * delta_x,
                        decimal_places)


def infer_binning(
    x: np.ndarray | list,
    atol: float = 1e-12,
) -> float:
    """
    Infers the coarsest bin width that is compatible with the given array.

    The returned value is the largest positive ``delta_x`` for which the
    finite values of ``x`` lie on a grid with spacing ``delta_x`` centered
    around zero, within the given tolerance.

    The function requires at least one value larger than the tolerance.

    Args:
        x:      List of decimal numbers whose binning should be inferred.
        atol:   Absolute tolerance used to suppress floating-point noise.

    Returns:
        delta_x:    Inferred coarsest compatible bin width.
    """
    if atol <= 0:
        raise ValueError("atol must be a positive number.")

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("The given array has no entry")
    if np.isnan(x).all():
        raise ValueError("The given array contains only NaN values")
    x = np.unique(x[~np.isnan(x)])

    decimal_places = abs(
        decimal.Decimal(str(atol)).as_tuple().exponent)
    quantum = decimal.Decimal(1).scaleb(-decimal_places)
    scale = 10 ** decimal_places
    quantized_values = [
        decimal.Decimal(str(value)).quantize(
            quantum,
            rounding=decimal.ROUND_HALF_UP,
        )
        for value in x
    ]

    gcd_scaled = 0
    for value in quantized_values:
        scaled_value = int(value * scale)
        gcd_scaled = math.gcd(gcd_scaled, abs(scaled_value))

    if gcd_scaled == 0:
        raise ValueError("Binning cannot be inferred from zero-only values.")

    delta_x = gcd_scaled / scale

    return float(delta_x)


def binning_test(
    x: np.ndarray | list,
    delta_x: float,
    atol: float = 1e-12,
    check_larger_binning: bool = True,
) -> bool:
    """
    Tests whether the given array is compatible with a bin width ``delta_x``.

    The function first checks whether the finite values of ``x`` lie on a grid
    with spacing ``delta_x`` centered around zero. If ``delta_x`` is zero or
    smaller than the given tolerance this compatibility check results allways
    in True.

    If ``check_larger_binning`` is False, that compatibility check is the final
    result. If it is True (default), the function additionally tests whether
    ``delta_x`` is the coarsest compatible bin width implied by the data.

    Args:
        x:                      List of decimal numbers that are supposedly
            binned (with bin-sizes ``delta_x``).
        delta_x:                Bin-size.
        atol:                   Absolute tolerance for the comparison. Default
            is 1e-12.
        check_larger_binning:   If True (default), the function not only checks
            that the binning of the array is compatible, but also makes sure
            that no larger compatible binning exists. In case of
            ``check_larger_binning = False``, the function only checks
            compatibility with the given ``delta_x``.

    Returns:
        result:     True if the array is binned to the given precision, False
                otherwise.

    Examples:
        >>> from seismostats.utils.binning import binning_test
        >>> binning_test([0.2,0.4,0.6], 0.2)
        True
        >>> binning_test([0.2,0.4,0.6], 0.1)
        False
        >>> binning_test([0.3,0.7,1.1], 0.4)
        False
        >>> binning_test([0.2,0.4,0.6], 0.05)
        False
        >>> binning_test([0.2,0.4,0.6], 0.1, check_larger_binning=False)
        True

    See also:
        :func:`~seismostats.utils.binning.bin_to_precision`
    """
    if delta_x < 0:
        raise ValueError("delta_x must be a non-negative number.")
    if atol < EPSILON:
        raise ValueError(f"atol must be >= {EPSILON}.")
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("The given array has no entry")
    if np.isnan(x).all():
        raise ValueError("The given array contains only NaN values")
    x = np.unique(x[~np.isnan(x)])
    all_zeros = np.all(np.abs(x) <= atol)
    if all_zeros:
        return True
    if delta_x >= atol:
        binned_x = bin_to_precision(x, delta_x)
        is_compatible = bool(np.allclose(
            x,
            binned_x,
            atol=atol,
            rtol=1e-16,
        ))
        if not is_compatible:
            return False
    if check_larger_binning is False:
        return True
    else:
        inferred_binning = infer_binning(x, atol=atol)
        return bool(np.isclose(
            inferred_binning,
            max(delta_x, atol),
            atol=atol,
            rtol=1e-16,
        ))


def get_fmd(
    magnitudes: np.ndarray,
    fmd_bin: float,
    weights: np.ndarray | None = None,
    bin_position: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates event counts per magnitude bin. Note that the returned bins
    array contains the center point of each bin unless
    ``bin_position = 'left'``.

    Args:
        magnitudes:     Array of magnitudes.
        fmd_bin:        Bin size for the FMD. This can be independent of
                    the discretization of the magnitudes. The optimal value
                    would be as small as possible while at the same time
                    ensuring that there are enough magnitudes in each bin.
        weights:        Array of weights for each magnitude.
        bin_position:   Position of the bin, options are  'center' and 'left'.
                    Accordingly, left edges of bins or center points are
                    returned.
    Returns:
        bins:           Array of bin centers (left to right).
        counts:         Counts for each bin.
        mags_binned:     Array of magnitudes binned to ``fmd_bin``.

    Examples:
        >>> from seismostats.utils import get_fmd

        >>> magnitudes = [0.9, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
        >>> fmd_bin = 1.0
        >>> bins, counts, mags_binned = get_fmd(magnitudes, fmd_bin)
        >>> bins
        array([1., 2.])
        >>> counts
        array([4, 3])
        >>> mags_binned
        array([1., 1., 1., 1., 2., 2., 2.])

    See also:
        :func:`~seismostats.utils.binning.get_cum_fmd`
    """
    magnitudes = np.asarray(magnitudes)
    if weights is not None:
        weights = np.asarray(weights)
    if fmd_bin <= 0:
        raise ValueError("Bin size (fmd_bin) must be a positive number.")

    mags_binned = bin_to_precision(magnitudes, fmd_bin)

    # use histogram to get the counts
    x_bins = bin_to_precision(
        np.arange(
            np.min(mags_binned), np.max(mags_binned) + 3 / 2 * fmd_bin, fmd_bin
        ),
        fmd_bin,
    )
    bins = x_bins[:-1].copy()
    x_bins -= fmd_bin / 2
    counts, _ = np.histogram(mags_binned, x_bins, weights=weights)

    assert (
        bin_position == "left" or bin_position == "center"
    ), "bin_position needs to be 'left' or 'center'"
    if bin_position == "left":
        bins = bins - fmd_bin / 2

    return bins, counts, mags_binned


def get_cum_fmd(
    magnitudes: np.ndarray,
    fmd_bin: float,
    weights: np.ndarray | None = None,
    bin_position: str = "center",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates cumulative event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless ``bin_position = 'left'`` is used.

    Args:
        magnitudes:           Array of magnitudes.
        fmd_bin:        Discretization of the magnitudes. It is possible to
                    provide a value that is larger than the actual
                    discretization of the magnitudes. In this case, the
                    magnitudes will be binned to the given ``fmd_bin``. This
                    might be useful for visualization purposes.
        weights:        Array of weights for each magnitude.
        bin_position:   Position of the bin, options are  'center' and 'left'.
                    Accordingly, left edges of bins or center points are
                    returned.

    Returns:
        bins:           Array of bin centers (left to right).
        c_counts:       Cumulative counts for each bin.
        mags_binned:           Array of magnitudes binned to ``fmd_bin``.

    Examples:
        >>> from seismostats.utils import get_cum_fmd

        >>> magnitudes = [0.9, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
        >>> fmd_bin = 1.0
        >>> bins, counts, mags_binned = get_cum_fmd(magnitudes, fmd_bin)
        >>> bins
        array([1., 2.])
        >>> counts
        array([7, 3])
        >>> mags_binned
        array([1., 1., 1., 2., 2., 2., 2.])

    See also:
        :func:`~seismostats.utils.binning.get_fmd`
    """
    magnitudes = np.asarray(magnitudes)
    if weights is not None:
        weights = np.asarray(weights)

    if fmd_bin == 0:
        mags_unique, bin_idx = np.unique(magnitudes, return_inverse=True)
        if weights is None:
            counts = np.bincount(bin_idx)
        else:
            counts = np.bincount(bin_idx, weights=weights)

        bins = mags_unique
        mags_binned = mags_unique[bin_idx]

    else:
        bins, counts, mags_binned = get_fmd(
            magnitudes,
            fmd_bin,
            weights=weights,
            bin_position=bin_position)
    c_counts = np.cumsum(counts[::-1])
    c_counts = c_counts[::-1]

    return bins, c_counts, mags_binned
