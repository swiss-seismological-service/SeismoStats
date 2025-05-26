import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

# Own functions
from seismostats.utils.binning import get_cum_fmd, get_fmd


def gutenberg_richter(
    magnitudes: np.ndarray, b_value: float, mc: float, n_mc: int
) -> np.ndarray:
    """
    Estimates the cumulative Gutenberg richter law (proportional to the
    complementary cumulative FMD) for a given magnitude vector.

    Args:
        magnitudes:     Array of magnitudes.
        b_value:        Theoretical b_value.
        mc:             Completeness magnitude.
        n_mc:           Total number of all events larger than the
                    completeness magnitude (:math: `n_{m_c} = 10^a`).

    Returns:
        x:  The cumulative Gutenberg-Richter distribution.
    """
    return n_mc * 10 ** (-b_value * (magnitudes - mc))


def plot_cum_fmd(
    magnitudes: np.ndarray | pd.Series,
    mc: float | None = None,
    delta_m: float = None,
    b_value: float | None = None,
    ax: plt.Axes | None = None,
    color: str | list = None,
    size: int = None,
    grid: bool = False,
    bin_position: str = "center",
    legend: bool | str | list = True,
) -> plt.Axes:
    """
    Plots cumulative frequency magnitude distribution, optionally with a
    corresponding theoretical Gutenberg-Richter (GR) distribution. The GR
    distribution is plotted provided the b-value is given.

    Args:
        magnitudes: Array of magnitudes.
        mc:         Completeness magnitude of the theoretical GR distribution.
        delta_m:    Discretization of the magnitudes; important for the correct
                visualization of the data. Assumed 0 if not given. It is
                possible to provide a value that is larger than the actual
                discretization of the magnitudes. In this case, the magnitudes
                will be binned to the given ``delta_m``.
        b_value:    The b-value of the theoretical GR distribution to plot.
        ax:         Axis where figure should be plotted.
        color:      Color of the data. If one value is given, it is used for
                points, and the line of the theoretical GR distribution if it
                is plotted. If a list of colors is given, the first entry is
                the color of the points, and the second of the line
                representing the GR distribution.
        size:       Size of the data points.
        grid:       Indicates whether or not to include grid lines.
        bin_position: Position of the bin, options are  'center' and 'left'
                accordingly, left edges of bins or center points are
                returned.

    Returns:
        ax: The ax object that was plotted on.
    """

    if delta_m is None:
        delta_m = 0

    magnitudes = magnitudes[~np.isnan(magnitudes)]
    bins, c_counts, magnitudes = get_cum_fmd(
        magnitudes, delta_m, bin_position=bin_position
    )

    if ax is None:
        ax = plt.subplots()[1]

    if type(legend) is list:
        labels = legend
    elif type(legend) is str:
        labels = [legend, "GR fit"]
    else:
        labels = ["cumulative", "GR fit"]

    if b_value is not None:
        if type(legend) is not list:
            labels[1] = "GR fit, b={x:.2f}".format(x=b_value)
        else:
            labels[1] = labels[1] + ", b={x:.2f}".format(x=b_value)
        if mc is None:
            mc = min(magnitudes)
        n_mc = len(magnitudes[magnitudes >= mc - delta_m / 2])
        if bin_position == "left":
            mc -= delta_m / 2
        x = bins[bins >= mc - delta_m / 2]
        y = gutenberg_richter(x, b_value, min(x), n_mc)

        if type(color) is not list:
            color = [color, color]

        ax.scatter(
            bins,
            c_counts,
            s=size,
            color=color[0],
            marker="s",
            label=labels[0],
        )
        ax.plot(x, y, color=color[1], label=labels[1])
    else:
        ax.scatter(
            bins,
            c_counts,
            s=size,
            color=color,
            marker="s",
            label=labels[0],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("N")

    if grid:
        ax.grid(True)
        ax.grid(which="minor", alpha=0.3)

    if legend:
        ax.legend()

    return ax


def plot_fmd(
    magnitudes: np.ndarray | pd.Series,
    fmd_bin: float,
    ax: plt.Axes | None = None,
    color: str = None,
    size: int = None,
    grid: bool = False,
    bin_position: str = "center",
    legend: bool | str | list = True,
) -> plt.Axes:
    """
    Plots frequency magnitude distribution.

    Args:
        magnitudes:     Array of magnitudes.
        fmd_bin:        Bin size for the FMD. This can be independent of
                    the descritization of the magnitudes. The optimal value
                    would be as small as possible while at the same time
                    ensuring that there are enough magnitudes in each bin.
        ax:             The axis where figure should be plotted.
        color:          Color of the data.
        size:           Size of data points.
        grid:           Indicates whether or not to include grid lines.
        bin_position:   Position of the bin, options are  "center" and "left"
                    accordingly, left edges of bins or center points are
                    returned.

    Returns:
        ax: The ax object that was plotted on.
    """

    magnitudes = magnitudes[~np.isnan(magnitudes)]

    bins, counts, magnitudes = get_fmd(
        magnitudes,
        fmd_bin,
        bin_position=bin_position
    )

    if ax is None:
        ax = plt.subplots()[1]

    if type(legend) is list:
        labels = legend
    elif type(legend) is str:
        labels = [legend]
    else:
        labels = ["non cumulative"]

    ax.scatter(bins, counts, s=size, color=color, marker="^", label=labels[0])
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("N")

    if grid:
        ax.grid(True)
        ax.grid(which="minor", alpha=0.3)

    if legend:
        ax.legend()

    return ax


def plot_cum_count(
    times: list | np.ndarray | pd.Series,
    magnitudes: np.ndarray | pd.Series,
    mcs: np.ndarray = np.array([0]),
    delta_m: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plots cumulative count of earthquakes in given catalog above given Mc
    through time. Plots a line for each given completeness magnitude in
    the array ``mcs``.

    Args:
        times:      Array containing times of events.
        magnitudes: Array of magnitudes of events corresponding to the
                ``times``.
        mcs:        The list of completeness magnitudes for which we show
                lines on the plot.
        delta_m:    Binning precision of the magnitudes, assumed 0 if not
                given.
        ax:         Axis where figure should be plotted.

    Returns:
        ax: Ax that was plotted on.
    """
    first_time, last_time = min(times), max(times)

    if ax is None:
        ax = plt.subplots()[1]

    if delta_m is None:
        delta_m = 0

    for mc in mcs:
        filtered_index = magnitudes >= mc - delta_m / 2
        times_sorted = sorted(times[filtered_index])
        times_adjusted = [first_time, *times_sorted, last_time]

        ax.plot(
            times_adjusted,
            np.arange(len(times_adjusted)) / (len(times_adjusted) - 1),
            label=f"Mc={np.round(mc, 2)}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative number of events")
    ax.legend()
    return ax


def plot_mags_in_time(
    times: list | np.ndarray | pd.Series,
    magnitudes: np.ndarray | pd.Series,
    mc_change_times: list | None = None,
    mcs: list | None = None,
    ax: plt.Axes | None = None,
    dot_smallest: int = 10,
    dot_largest: int = 200,
    dot_interpolation_power: int = 2,
    color_dots: str = "blue",
    color_line: str = "#eb4034",
) -> plt.Axes:
    """
    Creates a scatter plot, each dot is an event. Time shown on the x-axis,
    magnitude shown on the y-axis, but also reflected in the size of dots.

    Optionally, adds lines that represent the change in completeness magnitude.
    For example, ``mc_change_times = [2000, 2005]`` and ``mcs = [3.5, 3.0]``
    means that between 2000 and 2005, Mc is 3.5 and after 2005, Mc is 3.0.

    Args:
        times:      Array containing times of events.
        magnitudes: Array of magnitudes of events corresponding to the
                ``times``.
        ax:         Axis where figure should be plotted.
        mc_change_times: List of points in time when Mc changes, sorted in
                increasing order, can be given as a list of datetimes or
                integers (years).
        mcs:        Changed values of Mc at times given in ``mc_change_times``.
        dot_smallest: Smallest dot size for magnitude scaling.
        dot_largest: Largest dot size for magnitude scaling.
        dot_interpolation_power: Interpolation power for scaling.
        color_dots: Color of the dots representing the events.
        color_line: Color of the line representing the Mc changes.

    Returns:
        ax: ax that was plotted on
    """
    if ax is None:
        ax = plt.subplots()[1]

    ax.scatter(
        times,
        magnitudes,
        s=dot_size(
            magnitudes,
            smallest=dot_smallest,
            largest=dot_largest,
            interpolation_power=dot_interpolation_power,
        ),
        c=color_dots,
        linewidth=0.5,
        alpha=0.8,
        edgecolor="k",
    )

    if mc_change_times is not None and mcs is not None:
        if isinstance(mc_change_times[0], int):
            mc_change_times = [dt.datetime(x, 1, 1) for x in mc_change_times]

        mc_change_times.append(np.max(times))
        mcs.append(mcs[-1])
        ax.step(mc_change_times, mcs, where="post", c=color_line)

    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    return ax


def dot_size(
    magnitudes: np.ndarray,
    smallest: float = 10,
    largest: float = 200,
    interpolation_power: int = 1,
) -> np.ndarray:
    """
    Auxiliary function, computes dot sizes proportional to a given array of
    magnitudes.

    The dot sizes are computed using a power interpolation between the smallest
    and largest size, with the given interpolation power.

    Args:
        magnitudes:     Array of magnitudes, influencing size of the dots.
        smallest:       The size of the smallest dot, in pixels.
        largest:        The size of the largest dot, in pixels.
        interpolation_power: The power used to interpolate between the smallest
                    and largest size. A value of 1 results in a linear
                    interpolation, while larger values result in a more
                    "concave" curve.

    Returns:
        sizes:  The sizes of the dots, proportional to their magnitudes.
            The returned sizes are between ``smallest`` and ``largest``.
    """
    if largest <= smallest:
        print(
            "largest value is not larger than smallest, "
            "setting it to whatever I think is better"
        )
        largest = 50 * max(smallest, 2)
    smallest_mag = np.min(magnitudes)
    largest_mag = np.max(magnitudes)

    mag_norm = (magnitudes - smallest_mag) / (largest_mag - smallest_mag)
    mag_powered = np.power(mag_norm, interpolation_power)
    sizes = mag_powered * (largest - smallest) + smallest

    return sizes


def reverse_dot_size(
    sizes: np.ndarray,
    min_mag: float,
    max_mag: float,
    interpolation_power: int = 1,
) -> np.ndarray:
    """
    Computes magnitudes proportional to a given array of dot sizes.

    The magnitudes are computed by reversing the dot size calculation
    performed by the :func: `dot_size` function.

    Args:
        sizes:      A list containing the sizes of the dots.
        min_mag:    The minimum magnitude in the dataset.
        max_mag:    The maximum magnitude in the dataset.
        smallest:   The size of the smallest dot, in pixels.
        largest:    The size of the largest dot, in pixels.
        interpolation_power: The power used to interpolate between the smallest
                and largest size. A value of 1 results in a linear
                interpolation, while larger values result in a more "concave"
                curve.

    Returns:
        magnitudes: Array of magnitudes corresponding to the given dot sizes.
    """
    if interpolation_power == 0:
        raise ValueError("interpolation_power cannot be 0")

    size_norm = (sizes - min(sizes)) / (max(sizes) - min(sizes))
    size_powered = np.power(size_norm, 1 / interpolation_power)
    magnitudes = size_powered * (max_mag - min_mag) + min_mag
    return magnitudes
