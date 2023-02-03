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


def plot_cum_count(
    ax: plt.Axes,
    cat: pd.DataFrame,
    mcs: Optional[np.ndarray]=np.array([0]),
    delta_m: Optional[float]=0.1
):
    """
    Plots cumulative count of earthquakes in given catalog above given Mc
    through time. Plots a line for each given Mc.
    
    Args:
        ax: axis where figure should be plotted
        cat: catalog given as a pandas dataframe, should contain the column
             "magnitude" and  either "time" or "year"
        mcs: the list of completeness magnitudes for which we show lines on the plot
        delta_m: binning precision of the magnitudes
    """

    for mc in mcs:
        cat_above_mc = cat.query(f"magnitude>={mc-delta_m/2}")

        try:
            years = pd.to_datetime(cat_above_mc["time"]).dt.year
        except:
            try:
                years = cat_above_mc["year"]
            except:
                raise Exception("Dataframe should contain 'year' or 'time' column.")

        counts, bin_edges = np.histogram(years, 
                                         bins=np.arange(np.min(years), np.max(years), 1))
        cumulative_counts = np.cumsum(counts) / np.sum(counts)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centres, cumulative_counts, label=f"Mc={mc}")

    ax.set_xlabel("time")
    ax.set_ylabel("count - cumulative")
    ax.legend()


def plot_mags_in_time(
    ax: plt.Axes,
    cat: pd.DataFrame,
    years: Optional[list]=None,
    mcs: Optional[list]=None
):
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
    """

    try:
        cat_years = pd.to_datetime(cat["time"]).dt.year
    except:
        try:
            cat_years = cat_above_mc["year"]
        except:
            raise Exception("Dataframe should contain 'year' or 'time' column.")

    ax.scatter(cat_years, cat["magnitude"], cat["magnitude"]**2)

    if years is not None and mcs is not None:
        years.append(np.max(cat_years) + 1)
        mcs.append(mcs[-1])
        ax.step(years, mcs, where="post", c="black")

    ax.set_xlabel("time")
    ax.set_ylabel("magnitude")
    
    
def plot_in_space(
    ax: plt.Axes,
    cat: pd.DataFrame
):
    """
    Shows spatial distribution of the events from the given catalog, adding
    a map into the background.

    Args:
        ax: axis where figure should be plotted
        cat: catalog given as a pandas dataframe, should contain the columns
             "magnitude", "latitude" and "longitude"
    """

    min_lat = min(cat["latitude"])
    max_lat = max(cat["latitude"])
    min_lon = min(cat["longitude"])
    max_lon = max(cat["longitude"])

    #TODO: improve this part
    step_lat = 1 if (max_lat-min_lat)/20 < 1 else 10
    step_lon = 1 if (max_lon-min_lon)/20 < 1 else 10

    from mpl_toolkits.basemap import Basemap
    m = Basemap(resolution='i',
                llcrnrlon=min_lon,
                llcrnrlat=min_lat,
                urcrnrlon=max_lon,
                urcrnrlat=max_lat,
                epsg=4326)
    m.arcgisimage(service='World_Physical_Map', xpixels=10000, verbose=False)
    m.drawcountries()
    m.drawparallels(np.arange(np.ceil(min_lat), np.ceil(max_lat), step_lat),
                    labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(np.ceil(min_lon), np.ceil(max_lon), step_lon),
                    labels=[0, 0, 0, 1])

    m.scatter(cat["longitude"], cat["latitude"], np.exp(cat["magnitude"])/10, c=cat["magnitude"])

    plt.colorbar(orientation = 'horizontal', label="magnitude")
