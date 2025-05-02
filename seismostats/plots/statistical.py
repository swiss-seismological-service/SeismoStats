# standard
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
# statistical
from scipy.stats import norm

from seismostats.analysis.b_significant import b_significant_1D
# Own functions
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator


def plot_mc_vs_b(
    magnitudes: np.ndarray,
    mcs: np.ndarray,
    delta_m: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    confidence_interval: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes:     Magnitudes of the catalog.
        mcs:            Completeness magnitudes.
        delta_m:        Discretization of the magnitudes.
        method:         Method used for b-value estimation.
        confidence_interval: Confidence interval that should be plotted.
        ax:             Axis where figure should be plotted.
        color:          Color of the data.
        label:          Label of the data that will be put in the legend.
        **kwargs:       Additional parameters to be passed to the b-value
                    estimator.

    Returns:
        ax: ax that was plotted on
    """

    b_values = []
    b_errors = []
    estimator = b_method()

    for mc in mcs:
        estimator.calculate(
            magnitudes, mc=mc, delta_m=delta_m, **kwargs)
        b_values.append(estimator.b_value)
        b_errors.append(estimator.std)

    b_values = np.array(b_values)
    b_errors = np.array(b_errors)

    if ax is None:
        _, ax = plt.subplots()

    error_factor = norm.ppf((1 + confidence_interval) / 2)
    ax.plot(mcs, b_values, "-o", color=color, label=label)
    ax.fill_between(
        mcs,
        b_values - error_factor * b_errors,
        b_values + error_factor * b_errors,
        alpha=0.2,
        color=color,
        linewidth=0,
    )
    ax.set_xlabel("Completeness magnitude $m_c$")
    ax.set_ylabel("b-value")
    ax.grid(True)
    if label is not None:
        ax.legend()

    return ax


def plot_b_series_constant_nm(
    magnitudes: np.ndarray,
    delta_m: float,
    mc: float | np.ndarray,
    times: np.ndarray,
    n_m: int,
    min_num: float = 2,
    b_method: BValueEstimator = ClassicBValueEstimator,
    plot_technique: Literal['left', 'midpoint', 'right'] = 'right',
    x_variable: np.ndarray | None = None,
    confidence_level: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the b-values estimated from a running window of n_m magnitudes.

    Args:
        magnitudes:     Magnitudes of the events. If x_variable is None,
                    the magnitudes are assumed to be sorted in the dimension
                    of interest.
        delta_m:        Magnitude bin width.
        mc:             Completeness magnitude. If a single value is provided,
                    it is used for all magnitudes. Otherwise, the individual
                    completeness of each magnitude can be provided.
        times:          Times of the events.
        n_m:            Number of magnitudes in each partition.
        min_num:        Minimum number of events from which a b-value is
                    estimated. If the number of events is smaller, the b-value
                    is set to np.nan.
        b_method:       Method used to estimate the b-values.
        plot_technique: Technique where to plot the b-values with respect to
                    the x-variable. Options are 'left', 'midpoint', 'right'.
                    If set to 'right' (default), the b-value is plotted at the
                    right edge. For time series, this is the most common choice
                    as it avoids optical illusions of the b-value predicting
                    future seismicity.
        x_variable:     Values of the dimension of interest, along which the
                    b-values should be plotted. It should be a 1D array with
                    the same length as the magnitudes, e.g., the time of the
                    events. If None, the b-values are plotted against the
                    event index.
        confidence_level: Confidence level of the CI that should be plotted.
                    Default is 0.95 (i.e., the 95% confidence interval is
                    plotted).
        ax:             Axis where the plot should be plotted.
        color:          Color of the data.
        label:          Label of the data that will be put in the legend.
        **kwargs:       Additional parameters to be passed to the b-value
                    estimator.

    Returns:
        ax:         Ax that was plotted on.
    """
    # sanity checks and preparation
    magnitudes = np.array(magnitudes)
    times = np.array(times)
    if isinstance(mc, (float, int)):
        mc = np.ones(len(magnitudes)) * mc
    else:
        mc = np.array(mc)
    if n_m < min_num:
        raise ValueError("n_m cannot be smaller than min_num.")
    if len(magnitudes) != len(times):
        raise IndexError("Magnitudes and times must have the same length.")
    if x_variable is None:
        x_variable = np.arange(len(magnitudes))
    elif len(x_variable) != len(magnitudes):
        raise IndexError(
            "x_variable must have the same length as magnitudes.")
    else:
        x_variable = np.array(x_variable)
        idx_sort = np.argsort(x_variable)
        magnitudes = magnitudes[idx_sort]
        times = times[idx_sort]
        mc = mc[idx_sort]
        x_variable = x_variable[idx_sort]
    if ax is None:
        _, ax = plt.subplots()
    if plot_technique == 'left':
        idx_start = 0
    elif plot_technique == 'midpoint':
        idx_start = n_m // 2
    elif plot_technique == 'right':
        idx_start = n_m - 1
    else:
        raise ValueError(
            "plot_technique must be 'left', 'midpoint', or 'right'.")

    # filter magnitudes
    idx = magnitudes >= mc - delta_m / 2
    magnitudes = magnitudes[idx]
    times = times[idx]
    mc = mc[idx]
    x_variable = x_variable[idx]

    # estimation
    estimator = b_method()
    b_values = np.ones(len(magnitudes)) * np.nan
    std_bs = np.ones(len(magnitudes)) * np.nan
    for ii in range(len(magnitudes) - n_m + 1):
        # runnning window of n_m magnitudes
        mags_window = magnitudes[ii:ii + n_m]
        times_window = times[ii:ii + n_m]
        mc_window = mc[ii:ii + n_m]

        # sort the magnitudes by time
        idx = np.argsort(times_window)
        mags_window = mags_window[idx]
        mc_window = mc_window[idx]

        # estimate the b-value
        estimator.calculate(
            mags_window, mc=max(mc_window), delta_m=delta_m, **kwargs)
        if estimator.n < min_num:
            b_values[idx_start + ii] = np.nan
            std_bs[idx_start + ii] = np.nan
        else:
            b_values[idx_start + ii] = estimator.b_value
            std_bs[idx_start + ii] = estimator.std

    # plotting
    ax.plot(x_variable, b_values, color=color, label=label)
    error_factor = norm.ppf((1 + confidence_level) / 2)
    ax.fill_between(
        x_variable,
        b_values - error_factor * std_bs,
        b_values + error_factor * std_bs,
        alpha=0.2,
        color=_adjust_color_brightness(color, 0.7),
        linewidth=0,
    )

    ax.set_ylabel("b-value")
    if label is not None:
        ax.legend()
    return ax


def plot_b_significant_1D(
        magnitudes: np.ndarray,
        times: np.ndarray,
        mc: np.ndarray,
        delta_m: float,
        n_ms: np.ndarray | None = None,
        min_num: float = 2,
        b_method: BValueEstimator = ClassicBValueEstimator,
        x_variable: np.ndarray | None = None,
        p_threshold: float = 0.05,
        ax: plt.Axes | None = None,
        color: str = "blue",
        label: str | None = None,
        **kwargs,
):
    """
    Plots the mean autocorrelation (MAC) vs the number of magnitudes uased per
    sample, together with the chosen tresold. If the MAC is outside of the
    confidence interval, the null hypothesis of a constant b-value can be
    rejected.

    Source:
        Mirwald et. al. 2024, SRL, How to b-signicant when analysing b-value
        variations

    Args:
        magnitudes: Magnitudes of the events.
        times:      Times of the events.
        mc:         Completeness magnitude. If a single value is provided, it
            is used for all magnitudes. Otherwise, the individual completeness
            of each magnitude can be provided.
        delta_m:    Magnitude descretization.
        n_ms:       List of number of magnitudes used per sample. If None,
            the function will use an array of values that are increasing by
            10 within a range of reasonable values.
        min_num:    Minimum number of events from which a b-value is estimated.
        b_method:   Method to estimate the b-values.
        x_variable: Values of the dimension of interest, along which the
            b-values should be plotted. It should be a 1D array with the same
            length as the magnitudes, e.g., the time of the events. If None,
            the b-values are plotted against the event index.
        p_threshold: Threshold above which the null hypothesis of a constant
            b-value can be rejected.
        ax:         Axis where the plot should be plotted.
        color:      Color of the data.
        label:      Label of the data that will be put in the legend.
        **kwargs:   Additional keyword arguments for the b-value estimator.

    """
    # sanity checks and preparation
    magnitudes = np.array(magnitudes)
    times = np.array(times)

    if isinstance(mc, (float, int)):
        mc = np.ones(len(magnitudes)) * mc
    if x_variable is None:
        x_variable = np.arange(len(magnitudes))
    else:
        x_variable = np.array(x_variable)
        if len(x_variable) != len(magnitudes):
            raise ValueError(
                "x_variable must have the same length as magnitudes.")
        # sort in the dimension of interest (x-variable)
        srt = np.argsort(x_variable)
        magnitudes = magnitudes[srt]
        times = times[srt]
        x_variable = x_variable[srt]
    if ax is None:
        _, ax = plt.subplots()

    idx = magnitudes >= mc - delta_m / 2
    magnitudes = magnitudes[idx]
    times = times[idx]
    mc = mc[idx]

    if n_ms is None:
        n_ms = np.arange(20, len(magnitudes) / 25, 10).astype(int)

    # estimate the MAC for each n_m
    p = np.zeros(len(n_ms))
    mac = np.zeros(len(n_ms))
    mu_mac = np.zeros(len(n_ms))
    std_mac = np.zeros(len(n_ms))
    for ii, n_m in enumerate(n_ms):
        p[ii], mac[ii], mu_mac[ii], std_mac[ii] = b_significant_1D(
            magnitudes, mc, delta_m, times, n_m, min_num=min_num,
            method=b_method, **kwargs)

    # plot the results
    ax.plot(n_ms, mac, color=color, marker='o', label=label)
    std_factor = norm.ppf(1 - p_threshold / 2)
    ax.fill_between(n_ms,
                    mu_mac - std_factor * std_mac,
                    mu_mac + std_factor * std_mac,
                    color=_adjust_color_brightness(color, 0.7),
                    alpha=0.2,
                    linewidth=0)
    ax.plot(n_ms, mu_mac, color=_adjust_color_brightness(
        color, 1.3), linestyle='--')
    if any(p < p_threshold):
        ax.plot(n_ms[p < p_threshold], mac[p < p_threshold],
                'o', color='r')

    ax.set_xlabel('$n_m$')
    ax.set_ylabel('MAC')
    if label is not None:
        ax.legend()
    return ax


def _adjust_color_brightness(color, factor=1.2):
    """
    Adjusts the brightness of a given Matplotlib color.

    Args:
        color:  A valid Matplotlib color string (e.g., "blue", "#ff5733", "C1").
        factor: A float value that adjusts the brightness of the color. if < 1,
            the color is lightened; if > 1, the color is darkened.

    Returns:
        str: An adjusted hex color string.

    Examples:
        .. code-block:: python
        from seismostats.plots.statistical import adjust_color_brightness
        original_color = "blue"
        lighter_color = _adjust_color_brightness(original_color, factor=0.7)
    """

    # Convert color name or hex to RGB (values between 0 and 1)
    rgb = mcolors.to_rgb(color)

    if factor < 1:
        # Lighten: Move RGB values closer to 1 (white)
        adjusted_rgb = tuple(c + (1 - c) * (1 - factor) for c in rgb)
    else:
        # Darken: Scale down RGB values
        adjusted_rgb = tuple(c / factor for c in rgb)

    # Convert back to hex
    return mcolors.to_hex(adjusted_rgb)
