# standard
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

# statistical
from scipy.stats import norm

# Own functions
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator


def plot_mc_vs_b(
    magnitudes: np.ndarray,
    mcs: np.ndarray,
    delta_m: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    confidence_intvl: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
    label: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes: magnitudes of the catalog
        mcs:        completeness magnitudes (list or numpy array)
        delta_m:    discretization of the magnitudes
        method:     method used for b-value estimation
        confidence_intvl:   confidence interval that should be plotted
        ax:         axis where figure should be plotted
        color:      color of the data
        label:      label of the data that will be put in the legend
        **kwargs:   Additional keyword arguments for the b-value
                estimator.

    Returns:
        ax that was plotted on
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

    error_factor = norm.ppf((1 + confidence_intvl) / 2)
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
        mags: np.ndarray,
        delta_m: float,
        mc: np.ndarray,
        times: np.ndarray,
        n_m: int,
        min_num: float = 2,
        b_method: BValueEstimator = ClassicBValueEstimator,
        plot_technique: Literal['left', 'midpoint', 'right'] = 'right',
        x_variable: np.ndarray | None = None,
        confidence: float = 0.95,
        ax: plt.Axes | None = None,
        color: str = "blue",
        label: str | None = None,
        **kwargs,
) -> plt.Axes:
    """
    Plots the b-values estimated from a running window of n_m magnitudes.

    Args:
        mags:   magnitudes of the events. If x_variable is None, the magnitudes
            are assumed to be sorted in the dimension of interest.
        delta_m:    magnitude bin width
        mc:     completeness magnitude. If a single value is provided, it is
            used for all magnitudes. Otherwise, the individual completeness of
            each magnitude can be provided.
        times:  times of the events
        n_m:   number of magnitudes in each partition
        min_num:    minimum number of events from which a b-value is estimated.
            If the number of events is smaller, the b-value is set to np.nan
        b_method:   method to estimate the b-values
        plot_technique:    technique where to plot the b-values with respect to
            the x-variable. Options are 'left', 'midpoint', 'right'. If set to
            'right' (default), the b-value is plotted at the right edge. For
            time series, this is the most common choice, as it avoids optical
            illusions of the b-value predicting future seismicity.
        x_variable: values of the dimension of interest, along which the
            b-values should be plotted. It should be a 1D array with the same
            length as the magnitudes, e.g., the time of the events. If None,
            the b-values are plotted against the event index.
        confidence:    confidence interval that should be plotted. Default
            is 0.95 (i.e., the 95% confidence interval is plotted)
        ax:    axis where the plot should be plotted
        color: color of the data

    Returns:
        ax that was plotted on
    """

    if isinstance(mc, (float, int)):
        if min(mags) < mc:
            raise ValueError("The completeness magnitude is larger than the "
                             "smallest magnitude")
        mc = np.ones(len(mags)) * mc
    else:
        if any(mags < mc):
            raise ValueError("There are earthquakes below their respective "
                             "completeness magnitude")

    if n_m < min_num:
        raise ValueError("n_m cannot be smaller than min_num")

    if not isinstance(mags, np.ndarray):
        raise ValueError("mags must be an array")
    if not isinstance(times, np.ndarray):
        raise ValueError("times must be an array")
    if len(mags) != len(times):
        raise ValueError("mags and times must have the same length")

    if x_variable is None:
        x_variable = np.arange(len(mags))
    elif len(x_variable) != len(mags):
        raise ValueError(
            "x_variable must have the same length as magnitudes")
    else:
        idx_sort = np.argsort(x_variable)
        mags = mags[idx_sort]
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
            "plot_technique must be 'left', 'midpoint', or 'right'")

    # estimation
    estimator = b_method()
    b_values = np.ones(len(mags)) * np.nan
    std_bs = np.ones(len(mags)) * np.nan
    for ii in range(len(mags) - n_m + 1):
        # runnning window of n_m magnitudes
        mags_window = mags[ii:ii + n_m]
        times_window = times[ii:ii + n_m]

        # sort the magnitudes and times
        idx = np.argsort(times_window)
        mags_window = mags_window[idx]
        times_window = times_window[idx]

        # estimate the b-value
        estimator.calculate(mags_window, mc=mc[ii], delta_m=delta_m, **kwargs)
        if estimator.n < min_num:
            b_values[idx_start + ii] = np.nan
            std_bs[idx_start + ii] = np.nan
        else:
            b_values[idx_start + ii] = estimator.b_value
            std_bs[idx_start + ii] = estimator.std

    # plotting
    ax.plot(x_variable, b_values, color=color, label=label)
    error_factor = norm.ppf((1 + confidence) / 2)
    ax.fill_between(
        x_variable,
        b_values - error_factor * std_bs,
        b_values + error_factor * std_bs,
        alpha=0.2,
        color=color,
        linewidth=0,
    )

    if label is not None:
        ax.legend()
    return ax
