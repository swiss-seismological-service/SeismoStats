import numpy as np
import decimal


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


def bin_to_precision(x: np.ndarray, delta_x: float = 0.1) -> np.ndarray:
    """
    Rounds a float number x to a given precision. If precision not given,
    assumes 0.1 bin size

    Args:
        x: decimal number that needs to be rounded
        delta_x: size of the bin, optional

    Returns:
        Value rounded to the given precision.
    """

    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return np.round(normal_round_to_int(x / delta_x) * delta_x, decimal_places)


def bin_magnitudes(mags: np.ndarray, delta_m: float = 0.1) -> np.ndarray:
    """
    Rounds a list of float numbers to a given precision.
    If precision not given, assumes 0.1 bin size.

    Args:
        mags: list of values that need rounding
        delta_m: size of the bin, optional

    Returns:
        List of values rounded to the given precision.
    """

    rounded_array = bin_to_precision(mags, delta_m)
    return rounded_array
