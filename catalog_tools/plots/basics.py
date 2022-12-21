import numpy as np
import matplotlib.pyplot as plt
from typing import Union

# Own functions
from catalog_tools.utils.binning import bin_to_precision


def gutenberg_richter(magnitudes: np.ndarray, b_value: float,
                               mc: float, n_mc: int) -> np.ndarray:
    """ gives the number of the cumulative gutenberg-richter law for a given
    magnitude vector. GR-law: N(M) = 10 ** (a-b(M-mc)), here we set
    n_mc = 10 ** a

    Args:
        magnitudes: vector of magnitudes
        b_value: theoretical b_value
        mc: completeness magnitude
        n_mc: cumulative number of events until mc (starting from the largest
            magnitudes
    """
    return n_mc * 10 ** (-b_value * (magnitudes - mc))


def plot_cum_fmd(
        ax: plt.Axes,
        mags: np.ndarray,
        color: str = 'blue',
        b_value: Union[float, None] = None,
        mc: Union[float, None] = None,
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
