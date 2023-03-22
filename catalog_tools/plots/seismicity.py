# standard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# for map plotting
from shapely.geometry import Polygon
import cartopy
from cartopy.io import shapereader
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas

# typing
from typing import Optional, List, Union


# Own functions
from catalog_tools.plots.basics import dot_size


def plot_in_space(
        cat: pd.DataFrame,
        resolution: str = '110m',
        include_map: Optional[bool] = False,
        country: Optional[str] = None,
        colors: Optional[str] = None,
        dot_smallest: int = 10,
        dot_largest: int = 200,
        dot_interpolation_power: int = 2
) -> Union[cartopy.mpl.geoaxes.GeoAxes, plt.Axes]:
    """
     Function plots seismicity on the surface. If include_map is chosen True,
    a nice natural earth map is used, otherwise the seismicity is just
    plotted on a blank grid. In the latter case, the grid is stretched to
    according to the midpoint latitude.

    Args:
        cat: dataframe- needs to have latitude, longitude and depth as
            entries
        resolution: resolution of map, '10m', '50m' and '110m' available
        include_map: if True, seismicity will be plotted on natural earth
            map, otherwise it will be plotted on a blank grid.
        country: name of country, if None map will fit to data points
        colors: color of background. if None is chosen, is will be either
            white or standard natural earth colors.
        dot_smallest: smallest dot size for magnitude scaling
        dot_largest:largest dot size for magnitude scaling
        dot_interpolation_power: interpolation power for scaling

    Returns:
        GeoAxis object or normal axis object
    """

    if include_map is True:
        ax = plot_in_space_w_map(
            cat, resolution=resolution, country=country, colors=colors,
            dot_smallest=dot_smallest, dot_largest=dot_largest,
            dot_interpolation_power=dot_interpolation_power)
    else:
        ax = plot_in_space_wo_map(
            cat, resolution=resolution, country=country,
            dot_smallest=dot_smallest, dot_largest=dot_largest,
            dot_interpolation_power=dot_interpolation_power)

    return ax


def plot_in_space_wo_map(
        cat: pd.DataFrame,
        resolution: str = '110m',
        country: Optional[str] = None,
        colors: Optional[str] = None,
        dot_smallest: int = 10,
        dot_largest: int = 200,
        dot_interpolation_power: int = 2,
) -> plt.Axes:
    """
    Function plots seismicity on blanc grid
Args:
    cat: dataframe- needs to have latitude, longitude and depth as
        entries
    resolution: resolution of map, '10m', '50m' and '110m' available
    country: name of country, if None map will fit to data points
    colors: color of background. if None is chosen, is will be either
        white or standard natural earth colors.
    dot_smallest: smallest dot size for magnitude scaling
    dot_largest:largest dot size for magnitude scaling
    dot_interpolation_power: interpolation power for scaling

Returns:
    GeoAxis object
    """
    pass


def plot_in_space_w_map(
        cat: pd.DataFrame,
        resolution: str = '110m',
        country: Optional[str] = None,
        colors: Optional[str] = None,
        dot_smallest: int = 10,
        dot_largest: int = 200,
        dot_interpolation_power: int = 2,
) -> cartopy.mpl.geoaxes.GeoAxes:
    """
    Function plots seismicity on natural earth map

    Args:
        cat: dataframe- needs to have latitude, longitude and depth as
            entries
        resolution: resolution of map, '10m', '50m' and '110m' available
        country: name of country, if None map will fit to data points
        colors: color of background. if None is chosen, is will be either
            white or standard natural earth colors.
        dot_smallest: smallest dot size for magnitude scaling
        dot_largest:largest dot size for magnitude scaling
        dot_interpolation_power: interpolation power for scaling

    Returns:
        GeoAxis object
    """
    # request data for use by geopandas
    category: str = 'cultural'
    name: str = 'admin_0_countries'
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)
    if colors is not None:
        stamen_terrain = cimgt.Stamen('terrain-background',
                                      desired_tile_form="L")
        plt.set_cmap(colors)
    else:
        stamen_terrain = cimgt.Stamen('terrain-background')

    # projections that involved
    st_proj = stamen_terrain.crs  # projection used by Stamen images
    ll_proj = ccrs.PlateCarree()  # CRS for raw long/lat

    ax = plt.subplot(projection=st_proj)

    if country is not None:
        # create box around country
        poly = [df.loc[df['ADMIN'] == country]['geometry'].values[0]]
        pad_lat = abs(poly[0].bounds[0] - poly[0].bounds[2]) * 0.05
        pad_lon = abs(poly[0].bounds[1] - poly[0].bounds[3]) * 0.05
        exts = [poly[0].bounds[0] - pad_lat, poly[0].bounds[2] + pad_lat,
                poly[0].bounds[1] - pad_lon, poly[0].bounds[3] + pad_lon]
        msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
        msk_stm = st_proj.project_geometry(msk, ll_proj)
        ax.add_geometries(msk_stm, st_proj, facecolor='white', edgecolor='grey',
                          alpha=0.6)
    else:
        # create box around the data points
        pad_lat = abs(max(cat['latitude']) - min(cat['latitude'])) * 0.05
        pad_lon = abs(max(cat['longitude']) - min(cat['longitude'])) * 0.05
        exts = [min(cat['longitude']) - pad_lon,
                max(cat['longitude']) + pad_lon,
                min(cat['latitude']) - pad_lat,
                max(cat['latitude']) + pad_lat]

    ax.set_extent(exts, crs=ll_proj)
    ax.add_image(stamen_terrain, 8, alpha=0.6)

    # gridlines
    gl = ax.gridlines(crs=ll_proj, draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.bottom_labels = False
    gl.right_labels = False

    ax.scatter(
        cat["longitude"],
        cat["latitude"],
        c='blue',
        edgecolor='k',
        s=dot_size(cat["magnitude"], smallest=dot_smallest, largest=dot_largest,
                   interpolation_power=dot_interpolation_power),
        zorder=100,
        transform=ccrs.PlateCarree(),
        linewidth=0.5, alpha=0.8,
    )

    return ax


def rect_from_bound(xmin: float, xmax: float, ymin: float, ymax: float
                    ) -> List[tuple]:
    """
    Makes list of tuples for creating a rectangle polygon
    Args:
        xmin: minimum x value
        xmax: maximum x value
        ymin: minimum y value
        ymax: maximum y value

    Returns:
        list of (x,y)'s for a rectangle
    """
    xs = [xmax, xmin, xmin, xmax]
    ys = [ymax, ymax, ymin, ymin]
    return [(x, y) for x, y in zip(xs, ys)]
