import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union

# Own functions
from catalog_tools.utils.binning import bin_to_precision


def gutenberg_richter(magnitudes: np.ndarray, b_value: float,
                      mc: float, n_mc: int) -> np.ndarray:
    """ Estimates the cumulative Gutenberg richter law (proportional to the
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
        ax: Optional[plt.Axes] = None,
        b_value: Optional[float] = None,
        mc: Optional[float] = None,
        delta_m: float = 0,
        color: Union[str, list] = 'blue',
        size: int = 3,
        grid: bool = False
) -> plt.Axes:
    """ Plots cumulative frequency magnitude distribution, optionally with a
    corresponding theoretical Gutenberg-Richter (GR) distribution (using the
    provided b-value)

    Args:
        mags    : array of magnitudes
        ax      : axis where figure should be plotted
        b_value : b-value of the theoretical GR distribution to plot
        mc      : completeness magnitude of the theoretical GR distribution
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        color   : color of the data. If theoretical GR distribution should be
            colored differently this should be a list with two entries
        size    : size of scattered data
        grid    : bool, include grid lines or not

    Returns:
        ax that was plotted on
    """

    mags = mags[~np.isnan(mags)]

    mags_unique, counts = np.unique(mags, return_counts=True)
    idx = np.argsort(mags_unique)
    mags_unique = mags_unique[idx[::-1]]
    counts = counts[idx[::-1]]

    if ax is None:
        ax = plt.subplots()[1]

    if b_value is not None:
        if mc is None:
            mc = min(mags)
        x = mags[mags >= mc]
        y = gutenberg_richter(x, b_value, mc, len(x))
        if type(color) is not list:
            color = [color, color]
        ax.plot(x - delta_m / 2, y, color=color[1])
        ax.scatter(mags_unique - delta_m / 2, np.cumsum(counts), s=size,
                   color=color[0], marker='s')
    else:
        ax.scatter(mags_unique - delta_m / 2, np.cumsum(counts), s=size,
                   color=color, marker='s')

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')

    if grid is True:
        ax.grid(True)
        ax.grid(which='minor', alpha=0.3)

    return ax


def get_cum_fmd(
        mags: np.ndarray,
        delta_m: float,
        left: bool = False
) -> [np.ndarray, np.ndarray]:
    """ Calculates cumulative event counts across all magnitude units
    (summed from the right). Note that the returned bins array contains
    the center point of each bin unless left is True. Moreover, note
    that all bins except for the last are half-open in np.histogram
    (right edge not included).

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes
        left    : When True, left edges of bins are returned. When false,
                center points are returned.

    Returns:
        bins    : array of bin centers
        c_counts: cumulative counts for each bin
    """
    bins = bin_to_precision(np.arange((np.min(mags)) * 1000,
                                      (np.max(mags) + 2 * delta_m) * 1000,
                                      delta_m * 1000) / 1000, delta_m)
    counts = np.histogram(mags, bins)[0][::-1]
    bins = bins[:-1][::-1]
    c_counts = np.cumsum(counts)

    if left:
        pass
    else:
        bins = bins + delta_m / 2

    return bins, c_counts


def plot_cum_fmd_classic(
        mags: np.ndarray,
        ax: Optional[plt.Axes] = None,
        b_value: Optional[float] = None,
        mc: Optional[float] = None,
        delta_m: float = 0.1,
        color: Union[str, list] = 'blue',
        size: int = 3,
        grid: bool = False,
        left: bool = False
) -> plt.Axes:
    """ Plots cumulative frequency magnitude distribution, optionally with a
    corresponding theoretical Gutenberg-Richter (GR) distribution (using the
    provided b-value). Unlike plot_cum_fmd, plots values for all bins and
    requires binning.

    Args:
        mags    : array of magnitudes
        ax      : axis where figure should be plotted
        b_value : b-value of the theoretical GR distribution to plot
        mc      : completeness magnitude of the theoretical GR distribution
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        color   : color of the data. If theoretical GR distribution should be
            colored differently this should be a list with two entries
        size    : size of scattered data
        grid    : bool, include grid lines or not
        left    : When True, left edges of bins are returned. When false,
                center points are returned.

    Returns:
        ax that was plotted on
    """

    mags = mags[~np.isnan(mags)]
    mags = bin_to_precision(mags, delta_m)

    bins, c_counts = get_cum_fmd(mags, delta_m, left=left)

    if ax is None:
        ax = plt.subplots()[1]

    if b_value is not None:
        if mc is None:
            mc = min(mags)
        x = mags[mags >= mc]
        y = gutenberg_richter(x, b_value, mc, len(x))
        if type(color) is not list:
            color = [color, color]
	if left:
            ax.plot(x - delta_m / 2, y, color=color[1])
	else:
	    ax.plot(x, y, color=color[1])
        ax.scatter(bins, c_counts, s=size,
                   color=color[0], marker='s')
    else:
        ax.scatter(bins, c_counts, s=size,
                   color=color, marker='s')

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')

    if grid is True:
        ax.grid(True)
        ax.grid(which='minor', alpha=0.3)

    return ax


def plot_fmd(
        mags: np.ndarray,
        delta_m: float = 0,
        ax: Optional[plt.Axes] = None,
        color: str = 'blue',
        size: int = 5,
        grid: bool = False
) -> plt.Axes:
    """ Plots frequency magnitude distribution (non cumulative)

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        ax      : axis where figure should be plotted
        color   : color of the data
        size    : size of scattered data
        grid    : bool, include grid lines or not

    Returns:
        ax that was plotted on
    """

    if delta_m == 0:
        delta_m = 0.1
    mags = bin_to_precision(mags, delta_m)
    mags = np.array(mags)

    if ax is None:
        ax = plt.subplots()[1]

    mags_unique, counts = np.unique(mags, return_counts=True)

    ax.scatter(mags_unique, counts, s=size, color=color, marker='^')
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')

    if grid is True:
        ax.grid(True)
        ax.grid(which='minor', alpha=0.3)

    return ax


def plot_cum_count(
    cat: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    mcs: Optional[np.ndarray] = np.array([0]),
    delta_m: Optional[float] = 0.1,
) -> plt.Axes:
    """
    Plots cumulative count of earthquakes in given catalog above given Mc
    through time. Plots a line for each given Mc.

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
        cat_above_mc = cat.query(f"magnitude>={mc-delta_m/2}")
        times = sorted(cat_above_mc["time"])
        times_adjusted = [first_time, *times, last_time]

        ax.plot(times_adjusted,
                np.arange(len(times_adjusted)) / (len(times_adjusted) - 1),
                label=f"Mc={np.round(mc, 2)}")

    ax.set_xlabel("time")
    ax.set_ylabel("count - cumulative")
    ax.legend()
    return ax


def plot_mags_in_time(
    cat: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    years: Optional[list] = None,
    mcs: Optional[list] = None
) -> plt.Axes:
    """
    Creates a scatter plot, each dot is an event. Time shown on x-axis,
    magnitude shown on y-axis, but also in size of the dot.

    Optionally, adds lines that represent the change in completeness magnitude.
    For example, year_bins = [2000, 2005] and mcs = [3.5, 3.0] means that
    between 2000 and 2005, Mc is 3.5 and after 2005, Mc is 3.0.

    Args:
        ax: axis where figure should be plotted
        cat: catalog given as a pandas dataframe, should contain the column
             "magnitude" and  either "time" or "year"
        years: list of years when Mc changes, sorted in increasing order
        mcs: changed values of Mc at times given in 'years'

    Returns:
        ax that was plotted on
    """
    try:
        cat_years = pd.to_datetime(cat["time"])
    except KeyError:
        try:
            cat_years = cat["year"]
        except KeyError:
            raise Exception("Dataframe needs a 'year' or 'time' column.")

    if ax is None:
        ax = plt.subplots()[1]
    ax.scatter(cat_years, cat["magnitude"], cat["magnitude"]**2)

    if years is not None and mcs is not None:
        years.append(np.max(cat_years) + 1)
        mcs.append(mcs[-1])
        ax.step(years, mcs, where="post", c="black")

    ax.set_xlabel("time")
    ax.set_ylabel("magnitude")
    return ax


def dot_size(
        magnitudes: np.array, smallest: int = 10, largest: int = 200,
        interpolation_power: int = 1
) -> np.array:
    """Compute dot sizes proportional to a given array of magnitudes.

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
        The returned sizes are between `smallest` and `largest`.
    """
    if largest <= smallest:
        print(
            "largest value is not larger than smallest, "
            "setting it to whatever I think is better")
        largest = 50 * max(smallest, 2)
    smallest_mag = np.min(magnitudes)
    largest_mag = np.max(magnitudes)

    mag_norm = (magnitudes - smallest_mag) / (largest_mag - smallest_mag)
    mag_powered = np.power(mag_norm, interpolation_power)
    sizes = mag_powered * (largest - smallest) + smallest

    return sizes
