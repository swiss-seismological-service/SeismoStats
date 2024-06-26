"""This module contains functions for the estimation of a-value.
"""

import numpy as np
import datetime as dt
import warnings


def estimate_a(magnitudes: np.ndarray,
               mc: float | None = None,
               m0: float | None = None,
               b_value: float | None = None,
               delta_T: dt.timedelta | None = None,
               T: np.ndarray | None = None,
               ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.
    N  T/ delta_T = 10^(a - b * (m0 - mc)) (1)
    where N is the number of events with magnitude greater than m0, T is the
    time interval the magnitudes are observed, delta_T is the time interval
    for the estimation of the a-value.
    The original GR law was without the factor T/ delta_T - however, in many
    contexts this has proven useful so that a-values for the same region can be
    compared.

    Args:
        magnitudes: Magnitude sample
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        m0:         Magnitude for which the a-value is estimated. If None, the
                a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution
        delta_T:    Time interval for the estimation of the a-value. If None,
            delta_T is set to the whole interval of the times. Often, delta_T
            is set to 1 year.
        T:          Time interval the magnitudes are observed. Only needed if 
            delta_T is not None.

    Returns:
        a: a-value of the Gutenberg-Richter distribution
    """

    if mc is None:
        mc = magnitudes.min()

    if m0 is None:
        m0 = mc
    elif m0 != mc:
        if b_value is None:
            raise ValueError("b_value must be provided if m0 not equal to mc.")

    if magnitudes.min() < mc:
        warnings.warn(
            "Completeness magnitude is higher than the lowest magnitude."
            "Cutting the magnitudes to the completeness magnitude.")
        magnitudes = magnitudes[magnitudes >= mc]

    a = np.log10(len(magnitudes)) - b_value * (m0 - mc)

    if delta_T is not None:
        if T is None:
            raise ValueError("T must be provided if delta_T is given.")
        a = a + np.log(delta_T / T)

    return a
