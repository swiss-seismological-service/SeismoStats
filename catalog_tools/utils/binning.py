import numpy as np


def normal_round_to_int(x: float) -> int:
    """
    Rounds a float number x to the closest integer.

    Args:
        x: decimal number that needs to be rounded
    
    Returns:
        Rounded value of the given number.
    """

    sign = 1 if (x >= 0) else -1
    x = abs(x)

    if x - np.floor(x) < 0.5:
        return int(sign * np.floor(x))
    return int(sign * np.ceil(x))


def normal_round(x: float, n: int) -> float:
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
    
    
def bin_to_precision(x: float, delta_m: float = 0.1) -> float:
    """
    Rounds a float number x to a given precision. If precision not given,
    assumes 0.1 bin size

    Args:
        x: decimal number that needs to be rounded
        delta_m: size of the bin, optional
    
    Returns:
        Value rounded to the given precision.
    """

    return normal_round_to_int(x / delta_m) * delta_m


def bin_magnitudes(mags: list, delta_m: float = 0.1) -> list:
    """
    Rounds a list of float numbers to a given precision. 
    If precision not given, assumes 0.1 bin size.

    Args:
        mags: list of values that need rounding
        delta_m: size of the bin, optional
   
    Returns:
        List of values rounded to the given precision.
    """

    rounded_list = [bin_to_precision(m, delta_m) for m in mags]
    return rounded_list
