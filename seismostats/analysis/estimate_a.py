"""This module contains functions for the estimation of a-value.
"""

import numpy as np
import warnings
from seismostats.utils._config import get_option


def estimate_a(magnitudes: np.ndarray,
               mc: float | None = None,
               m_ref: float | None = None,
               b_value: float | None = None,
               T: float | None = None,
               ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.

    N = 10 ** (a - b * (m_ref - mc)) (1)

    where N is the number of events with magnitude greater than m_ref, which
    occurred in the time interval T. T should be given as a float- to begit 
    precise, it should be the time interval scaled to the time-unit of interest.
    E.g., if the number of events per year are of interest, T should be the
    number of years in which the events occurred.

    If only the magnitudes are given, the a-value is estimated at the lowest
    magnitude in the sample, with mc = min(magnitudes). Eq. (1) then simplifies
    to N = 10**a.

    Args:
        magnitudes: Magnitude sample
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution
        T:          Relative length of the time interval in which the events
                occurred (relative to the time unit of interest, e.g., years)

    Returns:
        a: a-value of the Gutenberg-Richter distribution
    """
    if mc is None:
        mc = magnitudes.min()
    elif magnitudes.min() < mc:
        if get_option("warnings") is True:
            warnings.warn(
                "Completeness magnitude is higher than the lowest magnitude."
                "Cutting the magnitudes to the completeness magnitude.")
        magnitudes = magnitudes[magnitudes >= mc]

    a = np.log10(len(magnitudes))

    # scale to reference magnitude
    if m_ref is not None:
        if b_value is None:
            raise ValueError(
                "b_value must be provided if m_ref is given")
        a = a - b_value * (m_ref - mc)

    # scale to reference time-interval
    if T is not None:
        a = a - np.log10(T)

    return a
