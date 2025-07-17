import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from seismostats.plots.seismicity import plot_in_space


def test_plot_in_space_legend():
    # Minimal test data
    lons = np.array([10, 11, 12])
    lats = np.array([50, 51, 52])
    mags = np.array([3.0, 4.0, 5.0])

    # Create a Cartopy GeoAxes with a PlateCarree projection
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Call your plotting function
    ax = plot_in_space(
        longitudes=lons,
        latitudes=lats,
        magnitudes=mags,
        ax=ax,
        include_map=False,
    )

    # Check legend
    legend = ax.get_legend()
    assert legend is not None, "Expected a legend but none was found."

    # Check legend title
    title = legend.get_title().get_text()
    assert title == "Magnitudes"

    # Check at least one label
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert len(legend_labels) > 0, "Expected at least one legend label."

    plt.close(fig)  # Close to avoid resource warnings
