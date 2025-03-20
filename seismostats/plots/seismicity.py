# standard
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
# for map plotting
from shapely.geometry import Polygon

# Own functions
from seismostats.plots.basics import dot_size, reverse_dot_size


def plot_in_space(
    longitudes: np.ndarray | pd.Series,
    latitudes: np.ndarray | pd.Series,
    magnitudes: np.ndarray | pd.Series,
    resolution: str = "10m",
    include_map: bool | None = False,
    country: str | None = None,
    colors: str | None = None,
    style: str = "satellite",
    dot_smallest: int = 10,
    dot_largest: int = 200,
    dot_interpolation_power: int = 2,
    dot_labels: str = "auto",
) -> cartopy.mpl.geoaxes.GeoAxes:
    """
    This function plots seismicity on a surface. If ``include_map`` is
    set to ``True``, a nice natural earth map is used, otherwise the seismicity
    is just plotted on a blank grid. In the latter case, the grid is stretched
    according to the midpoint latitude.

    Args:
        longitudes:     Array of longitudes.
        latitudes:      Array of latitudes.
        magnitudes:     Array of magnitudes, used for scaling of dot sizes.
        resolution:     Resolution of the map, "10m", "50m" and "110m"
                    available.
        include_map:    If True, seismicity will be plotted on natural earth
                    map, otherwise it will be plotted on a blank grid.
        country:        Name of country, if None map will fit to data points.
        colors:         Color of background. If None is chosen, it will be
                    either white or standard natural earth colors.
        style:          Style of map, "satellite" or "street" are available.
        dot_smallest:   Smallest dot size for magnitude scaling.
        dot_largest:    Largest dot size for magnitude scaling.
        dot_interpolation_power: Interpolation power for scaling.
        dot_labels:     Determines how labels for
                    magnitudes can be created. Input for matplotlib's
                    ``PathCollection.legend_elements``. If ``None``, no
                    label is shown. If an integer, target to use
                    ``dot_labels`` elements in the normed range.
                    If "auto", an automatic range is chosen for the
                    labels (default). If a list, uses elements of list
                    which are between minimum and maximum magnitude of
                    dataset for the legend.
                    Finally, a ``~.ticker.Locator`` can be provided to use
                    a predefined ``matplotlib.ticker`` (e.g.
                    ``FixedLocator``, which results in the same legend as
                    providing a list of values).
    Returns:
        ax: GeoAxis object
    """
    # request data for use by geopandas
    if include_map is True:
        category: str = "cultural"
        name: str = "admin_0_countries"
        shpfilename = shapereader.natural_earth(resolution, category, name)
        df = geopandas.read_file(shpfilename)

    if colors is not None:
        tiles = cimgt.GoogleTiles(style=style, desired_tile_form="L")
        plt.set_cmap(colors)
    else:
        tiles = cimgt.GoogleTiles(style=style)

    # projections that involved
    tile_proj = tiles.crs  # projection used by tiles
    ll_proj = ccrs.PlateCarree()  # CRS for raw long/lat

    ax = plt.subplot(projection=tile_proj)

    if include_map is True and country is not None:
        # create box around country
        poly = [df.loc[df["ADMIN"] == country]["geometry"].values[0]]
        pad_lat = abs(poly[0].bounds[0] - poly[0].bounds[2]) * 0.05
        pad_lon = abs(poly[0].bounds[1] - poly[0].bounds[3]) * 0.05
        exts = [
            poly[0].bounds[0] - pad_lat,
            poly[0].bounds[2] + pad_lat,
            poly[0].bounds[1] - pad_lon,
            poly[0].bounds[3] + pad_lon,
        ]
        msk = Polygon(_rect_from_bound(*exts)).difference(
            poly[0].simplify(0.01)
        )
        msk_stm = tile_proj.project_geometry(msk, ll_proj)
        ax.add_geometries(
            msk_stm, tile_proj, facecolor="white", edgecolor=None, alpha=0.6
        )
    else:
        # create box around the data points
        pad_lat = abs(max(latitudes) - min(latitudes)) * 0.05
        pad_lon = abs(max(longitudes) - min(longitudes)) * 0.05
        exts = [
            min(longitudes) - pad_lon,
            max(longitudes) + pad_lon,
            min(latitudes) - pad_lat,
            max(latitudes) + pad_lat,
        ]

    ax.set_extent(exts, crs=ll_proj)

    if include_map is True:
        ax.add_image(tiles, 8, alpha=0.6)

    # gridlines
    gl = ax.gridlines(
        crs=ll_proj,
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.bottom_labels = False
    gl.right_labels = False

    points = ax.scatter(
        longitudes,
        latitudes,
        c="blue",
        edgecolor="k",
        s=dot_size(
            magnitudes,
            smallest=dot_smallest,
            largest=dot_largest,
            interpolation_power=dot_interpolation_power,
        ),
        zorder=100,
        transform=ccrs.PlateCarree(),
        linewidth=0.5,
        alpha=0.8,
    )

    # insert legend
    if dot_interpolation_power == 0:  # all dots have the same size
        dot_labels = None
    if dot_labels is not None:
        if isinstance(dot_labels, np.ndarray):
            dot_labels = dot_labels.tolist()
        handles, labels = points.legend_elements(
            prop="sizes",
            num=dot_labels,
            c="blue",
            alpha=0.5,
            func=lambda x: reverse_dot_size(
                x,
                min(magnitudes),
                max(magnitudes),
                dot_interpolation_power,
            ),
        )
        ax.legend(handles, labels, title="Magnitudes")
    else:
        # no legend is shown
        pass

    return ax


def _rect_from_bound(
    xmin: float, xmax: float, ymin: float, ymax: float
) -> list[tuple]:
    """
    Makes list of tuples for creating a rectangle polygon
    Args:
        xmin:   Minimum x value.
        xmax:   Maximum x value.
        ymin:   Minimum y value.
        ymax:   Maximum y value.

    Returns:
        x: List of tuples representing the coordinates of vertices
        for a rectangle.
    """
    xs = [xmax, xmin, xmin, xmax]
    ys = [ymax, ymax, ymin, ymin]
    return [(x, y) for x, y in zip(xs, ys)]
