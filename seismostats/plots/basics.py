import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Own functions
from seismostats.utils.binning import get_cum_fmd, get_fmd


def gutenberg_richter(
    magnitudes: np.ndarray, b_value: float, mc: float, n_mc: int
) -> np.ndarray:
    """Estimates the cumulative Gutenberg richter law (proportional to the
    complementary cumulative FMD) for a given magnitude vector.

    Args:
        magnitudes: vector of magnitudes
        b_value: theoretical b_value
        mc: completeness magnitude
        n_mc: cumulative number of all events larger than the completeness
            magnitude (n_mc = 10 ** a)
    """
    return n_mc * 10 ** (-b_value * (magnitudes - mc))


def plot_cum_fmd(
    mags: np.ndarray,
    ax: plt.Axes | None = None,
    b_value: float | None = None,
    mc: float | None = None,
    delta_m: float = 0,
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
        mags    : array of magnitudes
        ax      : axis where figure should be plotted
        b_value : b-value of the theoretical GR distribution to plot
        mc      : completeness magnitude of the theoretical GR distribution
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        color   : color of the data. If one value is given, it is used for
            points, and the line of the theoretical GR distribution if it
            is plotted. If a list of colors is given, the first entry is
            the color of the points, and the second of the line representing
            the GR distribution.
        size    : size of data points
        grid    : whether to include grid lines or not
        bin_position    : position of the bin, options are  'center' and 'left'
                        accordingly, left edges of bins or center points are
                        returned.

    Returns:
        ax that was plotted on
    """

    mags = mags[~np.isnan(mags)]
    bins, c_counts, mags = get_cum_fmd(
        mags, delta_m, bin_position=bin_position
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
            mc = min(mags)
        if bin_position == "left":
            mc -= delta_m / 2

        n_mc = len(mags[mags >= mc])
        x = bins[bins >= mc]
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
    mags: np.ndarray,
    ax: plt.Axes | None = None,
    delta_m: float = 0.1,
    color: str = None,
    size: int = None,
    grid: bool = False,
    bin_position: str = "center",
    legend: bool | str | list = True,
) -> plt.Axes:
    """
    Plots frequency magnitude distribution. If no binning is specified, the
    assumed value of ``delta_m`` is 0.1.

    Args:
        mags    : array of magnitudes
        ax      : axis where figure should be plotted
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        color   : color of the data
        size    : size of data points
        grid    : whether to include grid lines or not
        bin_position    : position of the bin, options are  "center" and "left"
                        accordingly, left edges of bins or center points are
                        returned.

    Returns:
        ax that was plotted on
    """

    mags = mags[~np.isnan(mags)]

    if delta_m == 0:
        delta_m = 0.1

    bins, counts, mags = get_fmd(mags, delta_m, bin_position=bin_position)

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
    cat: pd.DataFrame,
    ax: plt.Axes | None = None,
    mcs: np.ndarray | None = np.ndarray([0]),
    delta_m: float | None = 0.1,
) -> plt.Axes:
    """
    Plots cumulative count of earthquakes in given catalog above given Mc
    through time. Plots a line for each given completeness magnitude in
    the array ``mcs``.

    Args:
        ax: axis where figure should be plotted
        cat: catalog given as a pandas dataframe, should contain the column
             "magnitude" and  either "time" or "year"
        mcs: the list of completeness magnitudes for which we show lines
             on the plot
        delta_m: binning precision of the magnitudes

    Returns:
        ax that was plotted on
    """
    try:
        times_list = cat["time"]
    except KeyError:
        raise Exception("Dataframe needs a 'time' column.")

    first_time, last_time = min(times_list), max(times_list)

    if ax is None:
        ax = plt.subplots()[1]

    for mc in mcs:
        cat_above_mc = cat.query(f"magnitude>={mc - delta_m / 2}")
        times = sorted(cat_above_mc["time"])
        times_adjusted = [first_time, *times, last_time]

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
    cat: pd.DataFrame,
    ax: plt.Axes | None = None,
    mc_change_times: list | None = None,
    mcs: list | None = None,
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
        ax: axis where figure should be plotted
        cat: catalog given as a pandas dataframe, should contain the column
             "magnitude" and  either "time" or "year"
        mc_change_times: list of points in time when Mc changes, sorted in
            increasing order, can be given as a list of datetimes or ints (yrs).
        mcs: changed values of Mc at times given in ``mc_change_times``
        dot_smallest: smallest dot size for magnitude scaling
        dot_largest: largest dot size for magnitude scaling
        dot_interpolation_power: interpolation power for scaling
        color_dots: color of the dots representing the events
        color_line: color of the line representing the Mc changes

    Returns:
        ax that was plotted on
    """
    year_only = False
    import datetime as dt

    try:
        times = pd.to_datetime(cat["time"])
    except KeyError:
        try:
            times = cat["year"]
            year_only = True
        except KeyError:
            raise Exception("Dataframe needs a 'year' or 'time' column.")

    if ax is None:
        ax = plt.subplots()[1]

    ax.scatter(
        times,
        cat["magnitude"],
        s=dot_size(
            cat["magnitude"],
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
        if not year_only and isinstance(mc_change_times[0], int):
            mc_change_times = [dt.datetime(x, 1, 1) for x in mc_change_times]
        if year_only and not isinstance(mc_change_times[0], int):
            mc_change_times = [x.year for x in mc_change_times]

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

    Args
    ----------
    magnitudes : array-like of float, shape (n_samples,)
        The magnitudes of the dots.
    smallest : float, optional (default=10)
        The size of the smallest dot, in pixels.
    largest : float, optional (default=200)
        The size of the largest dot, in pixels.
    interpolation_power : float, optional (default=1)
        The power used to interpolate between the smallest and largest size.
        A value of 1 results in a linear interpolation, while larger values
        result in a more "concave" curve.

    Returns
    -------
    sizes : ndarray of float, shape (n_samples,)
        The sizes of the dots, proportional to their magnitudes.
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
    """Compute magnitudes proportional to a given array of dot sizes.

    The magnitudes are computed by reversing the dot size calculation
    performed by the dot_size function.

    Args
    ----------
    sizes : array-like of float, shape (n_samples,)
        The sizes of the dots.
    min_mag : float
        The minimum magnitude in the dataset.
    max_mag : float
        The maximum magnitude in the dataset.
    smallest : float, optional (default=10)
        The size of the smallest dot, in pixels.
    largest : float, optional (default=200)
        The size of the largest dot, in pixels.
    interpolation_power : float, optional (default=1)
        The power used to interpolate between the smallest and largest size.
        A value of 1 results in a linear interpolation, while larger values
        result in a more "concave" curve.

    Returns
    -------
    magnitudes : ndarray of float, shape (n_samples,)
        The magnitudes corresponding to the given dot sizes.
    """
    if interpolation_power == 0:
        raise ValueError("interpolation_power cannot be 0")

    size_norm = (sizes - min(sizes)) / (max(sizes) - min(sizes))
    size_powered = np.power(size_norm, 1 / interpolation_power)
    magnitudes = size_powered * (max_mag - min_mag) + min_mag
    return magnitudes
