"""This module contains functions for the estimation of a-value.
"""

import numpy as np
import warnings
from seismostats.utils._config import get_option


def estimate_a(magnitudes: np.ndarray,
               mc: float | None = None,
               m_ref: float | None = None,
               b_value: float | None = None,
               scaling: float | None = None,
               ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.

    N = 10 ** (a - b * (m_ref - mc)) (1)

    where N is the number of events with magnitude greater than m_ref, which
    occurred in the region and time of interest. The parameter ``scaling`` can
    be used to normalize the number of events. E.g., if the number of events
    per year are of interest, ``scaling`` should be the number of years in
    which the events occurred. The scaling factor can also encompass other
    factors, such as the area of the region of interest.

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
        scaling:    Scaling factor. This should be chosen such that the number
                of events observed can be normalized, e.g., to the time and
                region of interest.

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

    # scale to reference time-interal or volume of interest
    if scaling is not None:
        a = a - np.log10(scaling)

    return a


def estimate_a_positive(
        magnitudes: np.ndarray,
        times: np.ndarray,
        delta_m: float = 0.1,
        mc: float | None = None,
        dmc: float | None = None,
        m_ref: float | None = None,
        b_value: float | None = None,
        scaling: float | None = None,
) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes whose magnitude m_i >= m_i-1 + dmc.

    Args:
        magnitudes: Magnitude sample
        times:      Times of the events. They should be sorted in ascending
                order. Also, it is important that each time is scaled to the
                time unit of interest (e.g., years). That is, in this case
                each time should be a float and represent the time in years.
        delta_m:    Discretization of the magnitudes
        dmc:        Minimum magnitude difference between consecutive events
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution. Only needed
                if m_ref is given.
        scaling:    Scaling factor. This should be chosen such that the number
                of events observed can be normalized, e.g., to the time and
                region of interest.

    Returns:
        a_pos: a-value of the Gutenberg-Richter distribution
    """

    if mc is None:
        mc = magnitudes.min()
    elif magnitudes.min() < mc:
        if get_option("warnings") is True:
            warnings.warn(
                "Completeness magnitude is higher than the lowest magnitude."
                "Cutting the magnitudes to the completeness magnitude.")
        idx = magnitudes >= mc
        magnitudes = magnitudes[idx]
        times = times[idx]

    if dmc is None:
        dmc = delta_m
    elif dmc < 0:
        raise ValueError("dmc must be larger or equal to 0")
    elif dmc < delta_m and get_option("warnings") is True:
        warnings.warn("dmc is smaller than delta_m, not recommended")

    # differences
    mag_diffs = np.diff(magnitudes)
    time_diffs = np.diff(times)
    if not np.all(time_diffs >= 0 * time_diffs[0]):
        raise ValueError("Times are not ordered correctly.")

    # only consider events with magnitude difference >= dmc
    idx = mag_diffs > dmc - delta_m / 2
    mag_diffs = mag_diffs[idx]
    time_diffs = time_diffs[idx]

    # estimate the number of events within the time interval
    total_time = times[-1] - times[0] + np.mean(np.diff(times))
    total_time_pos = sum(time_diffs) + np.mean(time_diffs)
    n_pos = total_time / total_time_pos * len(mag_diffs)

    # estimate a-value
    a = np.log10(n_pos)

    # scale to reference magnitude
    if m_ref is not None:
        if b_value is None:
            raise ValueError(
                "b_value must be provided if m_ref is given")
        a = a - b_value * (m_ref - mc)

    # scale to reference time-interal or volume of interest
    if scaling is not None:
        a = a - np.log10(scaling)

    return a
