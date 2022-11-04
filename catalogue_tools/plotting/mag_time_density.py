import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_magnitude_time_density(magnitudes, years, mbins, time_bins,
                                completeness_table=[], vmin=1, vmax=100,
                                filename=None, filetype="png", dpi=300):
    """
    Create a magnitude density plot
    :param magnitudes:
        Vector of magnitudes
    :param years:
        Vector of years
    :param mbins:
        Edges of the magnitude bins
    :param time_bins:
        Edges of the time bins
    :param completeness_table:
        If present, the table of completeness
    """
    # Generate a 2-D historgram in terms of magnitude and time
    counter = np.histogram2d(years, magnitudes, bins=(time_bins, mbins))[0]
    # Plot the density
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_bins[:-1], mbins[:-1],
                   counter.T, norm=LogNorm(vmin, vmax))
    # Add axes and labels
    plt.xlabel("Year", fontsize=16)
    plt.ylabel("Magnitude", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.colorbar()
    plt.grid()
    # If a completeness table is given add on a step line
    if len(completeness_table):
        completeness = np.array(completeness_table)
        plt.step(completeness[:, 0], completeness[:, 1], "k--", lw=2)
    # If the filename is specified then export to file
    if filename:
        plt.savefig(filename, format=filetype, dpi=300, bbox_inches="tight")
