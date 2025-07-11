import matplotlib.pyplot as plt
import numpy as np
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
    magnitudes: np.ndarray,
    mc: float | None = None,
    fmd_bin: float = None,
    weights: np.ndarray | None = None,
    b_value: float | None = None,
    ax: plt.Axes | None = None,
    color: str | None = None,
    color_line: str | None = None,
    size: int = None,
    grid: bool = False,
    bin_position: str = "center",
    label: bool | str = True,
    label_line: bool | str = True,
) -> plt.Axes:
    """
    Plots cumulative frequency magnitude distribution, optionally with a
    corresponding theoretical Gutenberg-Richter (GR) distribution. The GR
    distribution is plotted provided the b-value is given.

    Args:
        magnitudes: Array of magnitudes.
        mc:         Completeness magnitude of the theoretical GR distribution.
        fmd_bin:    Discretization of the magnitudes; important for the correct
                visualization of the data. Assumed 0 if not given. It is
                possible to provide a value that is larger than the actual
                discretization of the magnitudes.
        weights:    Weights for the magnitudes, defaults to None
        b_value:    The b-value of the theoretical GR distribution to plot.
        ax:         Axis where figure should be plotted.
        color:      Color of the data points. If None is chosen, it will be set
                to the default matplotlib color cycle.
        color_line: Color of the GR line, if None is chosen, it will be the
                same as the data points.
        size:       Size of the data points.
        grid:       Indicates whether or not to include grid lines.
        bin_position: Position of the bin, options are  'center' and 'left'
                accordingly, left edges of bins or center points are
                returned.
        label:     Label of the data points. If True, it will be set to
                "cumulative". If a string is provided, it will be used as the
                label for the data points. If False, no label is shown.
        label_line: If True, the GR line will be labeled as "GR fit", together
                with the provided b-value. If a string is provided, it will be
                used as the label for the GR line. If False, no label is shown.

    Returns:
        ax: The ax object that was plotted on.
    """

    if fmd_bin is None:
        fmd_bin = 0

    magnitudes = magnitudes[~np.isnan(magnitudes)]
    bins, c_counts, magnitudes = get_cum_fmd(
        magnitudes, fmd_bin, weights=weights, bin_position=bin_position
    )

    if ax is None:
        ax = plt.subplots()[1]

    if label is True:
        label = ["cumulative"]
    elif label is False:
        label = ''

    scatter1 = ax.scatter(
        bins,
        c_counts,
        s=size,
        color=color,
        marker="s",
        label=label,
    )

    if b_value is not None:
        if label_line is True:
            label_line = "GR fit, b={x:.2f}".format(x=b_value)
        elif label_line is False:
            label_line = ''
        if mc is None:
            mc = min(magnitudes)
        if color_line is None:
            color_line = scatter1.get_facecolor()[0]

        n_mc = len(magnitudes[magnitudes >= mc - fmd_bin / 2])
        if bin_position == "left":
            mc -= fmd_bin / 2
        x = bins[bins >= mc - fmd_bin / 2]
        y = gutenberg_richter(x, b_value, min(x), n_mc)

        ax.plot(x, y, color=color_line, label=label_line)

    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("N")

    if grid:
        ax.grid(True)
        ax.grid(which="minor", alpha=0.3)

    if label is not False or label_line is not False:
        ax.legend()

    return ax


def plot_fmd(
    magnitudes: np.ndarray,
    fmd_bin: float,
    weights: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    color: str = None,
    size: int = None,
    grid: bool = False,
    bin_position: str = "center",
    label: bool | str = True,
) -> plt.Axes:
    """
    Plots frequency magnitude distribution.

    Args:
        magnitudes:     Array of magnitudes.
        fmd_bin:        Bin size for the FMD. This can be independent of
                    the discretization of the magnitudes. The optimal value
                    would be as small as possible while at the same time
                    ensuring that there are enough magnitudes in each bin.
        weights:        Weights for the magnitudes, defaults to None
        ax:             The axis where figure should be plotted.
        color:          Color of the data.
        size:           Size of data points.
        grid:           Indicates whether or not to include grid lines.
        bin_position:   Position of the bin, options are  "center" and "left"
                    accordingly, left edges of bins or center points are
                    returned.
        label:          Label of the data points. If True, it will be set to
                    "non cumulative". If a string is provided, it will be used
                    as the label for the data points. If False, no label is
                    shown.

    Returns:
        ax: The ax object that was plotted on.
    """

    magnitudes = magnitudes[~np.isnan(magnitudes)]

    bins, counts, magnitudes = get_fmd(
        magnitudes,
        fmd_bin,
        weights=weights,
        bin_position=bin_position
    )

    if ax is None:
        ax = plt.subplots()[1]

    if label is True:
        label = ["non cumulative"]

    ax.scatter(bins, counts, s=size, color=color, marker="^", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("N")

    if grid:
        ax.grid(True)
        ax.grid(which="minor", alpha=0.3)

    if label is not False:
        ax.legend()

    return ax


def plot_cum_count(
    times: np.ndarray,
    magnitudes: np.ndarray,
    mcs: np.ndarray | None = None,
    color: str | list | None = None,
    delta_m: float = None,
    weights: np.ndarray | None = None,
    normalize: bool = True,
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
        color:     Color of the lines, corresponding to the different
                completeness magnitudes (if no completeness is given, the
                lowest magnitude of the catalog will be chosen as completeness).
                It should have the same length as ``mcs``. If there is only one
                completeness magnitude, it can be a single string. If None is
                chosen, it will be set to the default matplotlib color cycle.
        delta_m:    Binning precision of the magnitudes, assumed 0 if not
                given.
        weights:    Weights for the magnitudes, defaults to None
        normalize:  If True (default), the cumulative count is normalized to
                one. Otherwise, the absolute cumulative count is plotted.
        ax:         Axis where figure should be plotted.

    Returns:
        ax: Ax that was plotted on.
    """
    times = np.asarray(times)
    magnitudes = np.asarray(magnitudes)
    n_events = len(times)
    first_time, last_time = min(times), max(times)

    if delta_m is None:
        delta_m = 0
    if mcs is None:
        mcs = np.array([min(magnitudes)])
    if weights is None:
        weights = np.ones(n_events)
    else:
        weights = np.asarray(weights)
    if isinstance(color, str) or color is None:
        color = [color]
    else:
        if len(color) != len(mcs):
            raise ValueError(
                "Length of color list must match length of mcs list."
            )
    if ax is None:
        ax = plt.subplots()[1]

    for ii, mc in enumerate(mcs):
        # filter events below mc
        mask = magnitudes >= mc - delta_m / 2
        times_selected = times[mask]
        weights_selected = weights[mask]

        # sort by time
        order = np.argsort(times_selected)
        times_selected = times_selected[order]
        weights_selected = weights_selected[order]

        # estimate cumulative count
        cumulative = np.concatenate(
            ([0], np.cumsum(weights_selected), [weights_selected.sum()])
        )
        timeline = np.concatenate(
            ([first_time], times_selected, [last_time])
        )

        if normalize:
            cumulative = cumulative / cumulative[-1]

        ax.step(timeline, cumulative, where="post",
                label=f"Mc={mc:.2f}", color=color[ii])

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative number of events")
    ax.legend()
    return ax


def plot_mags_in_time(
    times: np.ndarray,
    magnitudes: np.ndarray,
    mc_change_times: list | None = None,
    mcs: list | None = None,
    ax: plt.Axes | None = None,
    dot_smallest: int = 10,
    dot_largest: int = 200,
    dot_interpolation_power: int = 2,
    color_dots: str | np.ndarray = "blue",
    cmap: str = "viridis",
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
        cmap:       Colormap for the dots, in case color_dots is an array.
                Default is "viridis".
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
        cmap=cmap,
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
