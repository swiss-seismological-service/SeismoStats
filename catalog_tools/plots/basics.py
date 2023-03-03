import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

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
        ax: plt.Axes,
        mags: np.ndarray,
        color: str = 'blue',
        b_value: Optional[float] = None,
        mc: Optional[float] = None,
        delta_m: float = 0):
    """ Plots cumulative frequency magnitude distribution, optionally with a
    corresponding theoretical Gutenberg-Richter (GR) distribution (using the
    provided b-value)

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        ax      : axis where figure should be plotted
        color   : color of the data

        b_value : b-value of the theoretical GR distribution to plot
        mc      : completeness magnitude of the theoretical GR distribution
    """
    mags_unique, counts = np.unique(mags, return_counts=True)
    idx = np.argsort(mags_unique)
    mags_unique = mags_unique[idx[::-1]]
    counts = counts[idx[::-1]]

    ax.scatter(mags_unique - delta_m / 2, np.cumsum(counts), 5, color=color)

    if b_value is not None:
        if mc is None:
            mc = min(mags)
        x = mags[mags >= mc]
        y = gutenberg_richter(x, b_value, mc, len(x))
        ax.plot(x - delta_m / 2, y, color=color)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')


def plot_fmd(ax: plt.Axes, mags: np.ndarray, color: str = 'blue',
             delta_m: float = 0):
    """ Plots frequency magnitude distribution (non cumulative)

    Args:
        mags    : array of magnitudes
        delta_m : discretization of the magnitudes, important for the correct
                visualization of the data
        ax      : axis where figure should be plotted
        color   : color of the data
    """

    if delta_m == 0:
        mags = bin_to_precision(mags, 0.1)
        mags = np.array(mags)

    mags_unique, counts = np.unique(mags, return_counts=True)
    ax.scatter(mags_unique, counts, 5, color=color)

    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('N')


def dot_size(magnitudes, smallest=10, largest=200, interpolation_power=1):
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
