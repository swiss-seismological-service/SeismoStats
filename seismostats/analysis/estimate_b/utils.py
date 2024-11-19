import numpy as np


def beta_to_b_value(beta: float) -> float:
    """converts the beta value to the b-value  of the Gutenberg-Richter law

    Args:
        beta: beta value

    Returns:
        b_value: corresponding b-value
    """
    return beta / np.log(10)


def b_value_to_beta(b_value: float) -> float:
    """converts the b-value to the beta value of the exponential distribution

    Args:
        b_value: b-value

    Returns:
        beta: corresponding beta value
    """
    return b_value * np.log(10)
