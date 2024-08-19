"""This module contains functions for the estimation of a-value.
"""

import numpy as np
import warnings
from seismostats.utils._config import get_option


def estimate_a(magnitudes: np.ndarray,
               times=None,
               mc: float | None = None,
               delta_m: float = None,
               m_ref: float | None = None,
               b_value: float | None = None,
               scaling_factor: float | None = None,
               method: str = "classic",
               ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.

    Args:
        magnitudes: Magnitude sample
        times:      Times of the events, in any format (datetime, float, etc.)
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        delta_m:    Discretization of the magnitudes.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution. Only needed
                if m_ref is given.
        scaling_factor:    Scaling factor. This should be chosen such that the
                number of events observed can be normalized. For example:
                Relative length of the time interval in which the events
                occurred (relative to the time unit of interest, e.g., years).
        method:     Method to estimate the a-value. Options are "classic" and
                "positive".

    """
    if method == "classic":
        return estimate_a_classic(
            magnitudes, mc, delta_m, m_ref, b_value, scaling_factor)
    elif method == "positive":
        return estimate_a_positive(
            magnitudes,
            times,
            mc,
            delta_m,
            m_ref,
            b_value,
            scaling_factor,
            correction=False)


def estimate_a_classic(magnitudes: np.ndarray,
                       mc: float | None = None,
                       delta_m: float = None,
                       m_ref: float | None = None,
                       b_value: float | None = None,
                       scaling_factor: float | None = None,
                       ) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law.

    N = 10 ** (a - b * (m_ref - mc)) (1)

    where N is the number of events with magnitude greater than m_ref, which
    occurred in the timeframe of the catalogue.

    If only the magnitudes are given, the a-value is estimated at the lowest
    magnitude in the sample, with mc = min(magnitudes). Eq. (1) then simplifies
    to N = 10**a.

    Args:
        magnitudes: Magnitude sample
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        delta_m:    Discretization of the magnitudes. This is needed solely to
                avoid rounding errors. By default rounding errors are not
                considered. This is adequate if the megnitudes are either
                coninuous or do not contain rounding errors.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution
        scaling factor. This should be chosen such that the
                number of events observed can be normalized. For example:
                Relative length of the time interval in which the events
                occurred (relative to the time unit of interest, e.g., years).

    Returns:
        a: a-value of the Gutenberg-Richter distribution
    """
    if delta_m is None:
        delta_m = 0
    if mc is None:
        mc = magnitudes.min()
    elif magnitudes.min() < mc - delta_m / 2:
        if get_option("warnings") is True:
            warnings.warn(
                "Completeness magnitude is higher than the lowest magnitude."
                "Cutting the magnitudes to the completeness magnitude.")
        magnitudes = magnitudes[magnitudes >= mc - delta_m / 2]

    a = np.log10(len(magnitudes))

    # scale to reference magnitude
    if m_ref is not None:
        if b_value is None:
            raise ValueError(
                "b_value must be provided if m_ref is given")
        a = a - b_value * (m_ref - mc)

    # scale to reference time-interal or volume of interest
    if scaling_factor is not None:
        a = a - np.log10(scaling_factor)

    return a


def estimate_a_positive(
        magnitudes: np.ndarray,
        times: np.ndarray,
        delta_m: float,
        mc: float | None = None,
        dmc: float | None = None,
        m_ref: float | None = None,
        b_value: float | None = None,
        scaling_factor: float | None = None,
        correction: bool = False,
) -> float:
    """Estimate the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes whose magnitude m_i >= m_i-1 + dmc.

    Args:
        magnitudes: Magnitude sample
        times:      Times of the events. They should be sorted in ascending
                order. Also, it is important that each time is scaled to the
                time unit of interest (e.g., years). That is, in this case
                each time should be a float and represent the time in years.
        delta_m:    Discretization of the magnitudes.
        dmc:        Minimum magnitude difference between consecutive events.
                If None, the default value is delta_m.
        mc:         Completeness magnitude. If None, the lowest magnitude is
                used as completeness magnitude.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, the a-value is estimated at mc.
        b_value:    b-value of the Gutenberg-Richter distribution. Only needed
                if m_ref is given.
        scaling factor. This should be chosen such that the
                number of events observed can be normalized. For example:
                Relative length of the time interval in which the events
                occurred (relative to the time unit of interest, e.g., years).
        correction: If True, the a-value is corrected for the bias introduced
                by the observation period being larger than the time interval
                between the first and last event. This is only relevant if the
                sample is very small (<30). The assumption is that the blind
                time in the beginning and end of the catalogue has a similar
                distribution as the rest of the catalogue. We think that this
                improves the estimate of the a-value for small samples, however,
                without proof.

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
        idx = magnitudes >= mc - delta_m / 2
        magnitudes = magnitudes[idx]
        times = times[idx]
    # TODO: check for correct binning

    if dmc is None:
        dmc = delta_m
    elif dmc < 0:
        raise ValueError("dmc must be larger or equal to 0")
    elif dmc < delta_m and get_option("warnings") is True:
        warnings.warn("dmc is smaller than delta_m, not recommended")

    # order the magnitudes and times
    idx = np.argsort(times)
    magnitudes = magnitudes[idx]
    times = times[idx]

    # differences
    mag_diffs = np.diff(magnitudes)
    time_diffs = np.diff(times)

    # only consider events with magnitude difference >= dmc
    idx = mag_diffs > dmc - delta_m / 2
    mag_diffs = mag_diffs[idx]
    time_diffs = time_diffs[idx]

    # estimate the number of events within the time interval
    total_time = times[-1] - times[0]
    total_time_pos = sum(time_diffs / total_time)
    if correction:
        total_time += 2 * np.mean(np.diff(times) / total_time)
        total_time_pos += np.mean(time_diffs / total_time)
    n_pos = 1 / total_time_pos * len(mag_diffs)

    # estimate a-value
    a = np.log10(n_pos)

    # scale to reference magnitude
    if m_ref is not None:
        if b_value is None:
            raise ValueError(
                "b_value must be provided if m_ref is given")
        a = a - b_value * (m_ref - mc)

    # scale to reference time-interal or volume of interest
    if scaling_factor is not None:
        a = a - np.log10(scaling_factor)

    return a
