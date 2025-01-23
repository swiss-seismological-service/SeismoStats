"""This module contains functions for the estimation of a-value.
"""

import warnings

import numpy as np

from seismostats.utils._config import get_option
from seismostats.utils.binning import binning_test
from seismostats.analysis.bvalue.utils import find_next_larger


def estimate_a(magnitudes: np.ndarray,
               scaling_factor: float | None = None,
               m_ref: float | None = None,
               mc: float | None = None,
               b_value: float | None = None,
               delta_m: float = None,
               times=None,
               method: str = "classic",
               ) -> float:
    r"""Return the a-value of the Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \cdot (m - m_{ref})}

    where N(m) is the number of events with magnitude larger or equal to m
    that occurred in the timeframe of the catalog.

    Args:
        magnitudes: vector of magnitudes, unsorted
        scaling_factor:    scaling factor. If given, this is used to normalize
                the number of observed events. For example:
                Volume or area of the region considered
                or length of the time interval, given in the unit of interest.
        m_ref:      reference magnitude for which the a-value is estimated. If
                None, m_ref is set to the smallest magnitude in the catalog.
        mc:         completeness magnitude.
                If given, magnitudes below mc are disregarded.
                If None, the catalog is assumed to be complete and
                mc is set to the smallest magnitude in the catalog.
                This is only relevant when m_ref is not None.
        b_value:    b-value of the Gutenberg-Richter law. Only relevant
                when m_ref is not None.
        delta_m:    discretization of magnitudes. default is no discretization
        times:      vector of times of the events, in any format (datetime,
                float, etc.) Only needed when method is "positive".
        method:     Method to estimate the a-value. Options are "classic" and
                "positive".

        Returns:
        a: a-value of the Gutenberg-Richter law

    """
    if method == "classic":
        return estimate_a_classic(
            magnitudes, scaling_factor, m_ref, mc, b_value, delta_m)
    elif method == "positive":
        return estimate_a_positive(
            magnitudes,
            times,
            delta_m,
            scaling_factor=scaling_factor,
            m_ref=m_ref,
            mc=mc,
            b_value=b_value)


def estimate_a_classic(magnitudes: np.ndarray,
                       scaling_factor: float | None = None,
                       m_ref: float | None = None,
                       mc: float | None = None,
                       b_value: float | None = None,
                       delta_m: float = None,
                       ) -> float:
    r"""Return the a-value of the Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \cdot (m - m_{ref})}

    where N(m) is the number of events with magnitude larger or equal to m
    that occurred in the timeframe of the catalog.

    Args:
        magnitudes: vector of magnitudes, unsorted
        scaling_factor:    scaling factor. If given, this is used to normalize
                the number of observed events. For example:
                Volume or area of the region considered
                or length of the time interval, given in the unit of interest.
        m_ref:      reference magnitude for which the a-value is estimated. If
                None, m_ref is set to the smallest magnitude in the catalog.
        mc:         completeness magnitude.
                If given, magnitudes below mc are disregarded.
                If None, the catalog is assumed to be complete and
                mc is set to the smallest magnitude in the catalog.
                This is only relevant when m_ref is not None.
        b_value:    b-value of the Gutenberg-Richter law. Only relevant
                when m_ref is not None.
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        a: a-value of the Gutenberg-Richter law
    """
    if delta_m is None:
        delta_m = 0
        tolerance = 1e-08
    else:
        tolerance = max(delta_m / 100, 1e-08)
    assert (
        binning_test(magnitudes, delta_m, tolerance, fail_if_larger=False)
    )
    "Magnitudes are not binned correctly."
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
        dmc: float | None = None,
        scaling_factor: float | None = None,
        m_ref: float | None = None,
        mc: float | None = None,
        b_value: float | None = None,
) -> float:
    """Return the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes with magnitude m_i >= m_i-1 + dmc.

    Source:
        Following the idea of positivity of
        van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2).
        Note: This is *not* a-positive as defined by van der Elst and Page 2023
        (JGR: Solid Earth, Vol 128, Issue 10).

    Args:
        magnitudes: vector of magnitudes, unsorted
        times:      vector of times of the events, in any format (datetime,
                float, etc.)
        delta_m:    discretization of the magnitudes.
                default is no discretization
        dmc:        minimum magnitude difference between consecutive events.
                If None, the default value is delta_m.
        scaling_factor:    scaling factor. If given, this is used to normalize
                the number of observed events. For example:
                Volume or area of the region considered
                or length of the time interval, given in the unit of interest.
        m_ref:      reference magnitude for which the a-value is estimated. If
                None, m_ref is set to the smallest magnitude in the catalog.
        mc:         completeness magnitude.
                If given, magnitudes below mc are disregarded.
                If None, the catalog is assumed to be complete and
                mc is set to the smallest magnitude in the catalog.
                This is only relevant when m_ref is not None.
        b_value:    b-value of the Gutenberg-Richter law. Only relevant
                when m_ref is not None.

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

    # test that the magnitudes are binned correctly
    if delta_m == 0:
        tolerance = 1e-08
    else:
        tolerance = max(delta_m / 100, 1e-08)
    assert (
        binning_test(magnitudes, delta_m, tolerance, fail_if_larger=False)
    )
    "Magnitudes are not binned correctly."

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

    time_factor = sum(time_diffs / total_time)
    n_pos = sum(idx) / time_factor

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


def estimate_a_more_positive(
        magnitudes: np.ndarray,
        times: np.ndarray,
        delta_m: float,
        b_value: float,
        dmc: float | None = None,
        scaling_factor: float | None = None,
        m_ref: float | None = None,
        mc: float | None = None,
) -> float:
    """Return the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes with magnitude m_i >= m_i-1 + dmc.

    Source:
        van der Elst and Page 2023 (JGR: Solid Earth, Vol 128, Issue 10).

    Args:
        magnitudes: vector of magnitudes, unsorted
        times:      vector of times of the events, in any format (datetime,
                float, etc.)
        delta_m:    discretization of the magnitudes.
        b_value:    b-value of the Gutenberg-Richter law.
        dmc:        minimum magnitude difference between consecutive events.
                If None, the default value is delta_m.
        scaling_factor:    scaling factor. If given, this is used to normalize
                the number of observed events. For example:
                Volume or area of the region considered
                or length of the time interval, given in the unit of interest.
        m_ref:      Reference magnitude for which the a-value is estimated. If
                None, m_ref is set to the smallest magnitude in the catalog.
        mc:         completeness magnitude.
                If given, magnitudes below mc are disregarded.
                If None, the catalog is assumed to be complete and
                mc is set to the smallest magnitude in the catalog.
                This is only relevant when m_ref is not None.

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
        idx = magnitudes > mc - delta_m / 2
        magnitudes = magnitudes[idx]
        times = times[idx]

    # test that the magnitudes are binned correctly
    if delta_m == 0:
        tolerance = 1e-08
    else:
        tolerance = max(delta_m / 100, 1e-08)
    assert (
        binning_test(magnitudes, delta_m, tolerance, fail_if_larger=False)
    )
    "Magnitudes are not binned correctly."

    if dmc is None:
        dmc = delta_m
    elif dmc < 0:
        raise ValueError("dmc must be larger or equal to 0")
    elif dmc < delta_m and get_option("warnings") is True:
        warnings.warn("dmc is smaller than delta_m, not recommended")

    if b_value is None:
        raise ValueError("b_value must be provided")

    # order the magnitudes and times
    idx = np.argsort(times)
    magnitudes = magnitudes[idx]
    times = times[idx]

    # differences
    idx_next_larger = find_next_larger(magnitudes, delta_m, dmc)
    time_diffs = times[idx_next_larger] - times

    # deal with events which do not have a next larger event
    idx_no_next = idx_next_larger == 0
    time_diffs[idx_no_next] = times[-1] - times[idx_no_next]

    # estimate the number of events within the time interval
    total_time = times[-1] - times[0]

    # scale the time
    tau = time_diffs * 10**(-b_value * (magnitudes + dmc - mc))

    time_factor = sum(tau / total_time)
    n_more_pos = sum(~idx_no_next) / time_factor

    # estimate a-value
    a = np.log10(n_more_pos)

    # scale to reference magnitude
    if m_ref is not None:
        a = a - b_value * (m_ref - mc)

    # scale to reference time-interal or volume of interest
    if scaling_factor is not None:
        a = a - np.log10(scaling_factor)

    return a
