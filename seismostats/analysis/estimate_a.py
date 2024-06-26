"""This module contains functions for the estimation of a-value.
"""

import numpy as np
import datetime as dt
import warnings


def estimate_a(magnitudes: np.ndarray,
               mc: float | None = None,
               m0: float | None = None,
               b_value: float | None = None,
               T: dt.timedelta | None = None,
               delta_T: dt.timedelta | None = None,
               ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.
    N = 10^(a - b * (m - mc)) (1)
    where N is the number of events with magnitude greater than m0, T is the
    time interval the magnitudes are observed, delta_T is the time interval
    for the estimation of the a-value.
    In order to make comparability across different time intervals possible, 
    the a-value is often estimated at a reference magnitude m0 and a reference
    time interval delta_T. Then, instead of (1), the following equation is used:
    N T/ delta_T = 10^(a - b * (m0 - mc)) (2)
    If only the magnitudes are provided, the a value is estimated at the lowest
    magnitude of the sample, using eq. (1). Otherwise, the a-value is estimated
    at the reference magnitude m0 and the reference time interval delta_T using
    eq. (2).

    Args:
        magnitudes: Magnitude sample
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        m0:         Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution
        T:          Reference time interval for the estimation of the a-value.
                If None, T is set to the whole interval of the times. Often, T
                is set to 1 year.
        delta_T:    Time interval which the magnitudes are observed in. Only
                needed if T is not None.

    Returns:
        a: a-value of the Gutenberg-Richter distribution
    """
    if mc is None:
        mc = magnitudes.min()
    elif magnitudes.min() < mc:
        warnings.warn(
            "Completeness magnitude is higher than the lowest magnitude."
            "Cutting the magnitudes to the completeness magnitude.")
        magnitudes = magnitudes[magnitudes >= mc]

    a = np.log10(len(magnitudes))

    # scale to reference magnitude
    if m0 is not None:
        if b_value is None:
            raise ValueError(
                "b_value must be provided if m0 is given")
        a = a - b_value * (m0 - mc)

    # scale to reference time-interval
    if T is not None:
        if delta_T is None:
            raise ValueError("T must be provided if delta_T is given.")
        a = a + np.log10(T / delta_T)

    return a
