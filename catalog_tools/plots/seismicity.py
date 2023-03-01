# standard
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
from typing import Optional
from typing import List


def plot_in_space(
        cat: pd.DataFrame,
        country: Optional[str] = None,
        resolution: str = '110m',
        colors: Optional[str] = None
) -> cartopy.mpl.geoaxes.GeoAxes:
    """
    function plots
    Args:
        cat: pd.DataFrame
        country: name of country, if None map will fit to data points
        resolution: resolution of map, '10m', '50m' and '110m' available

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
        exts = [min(cat['latitude']) - pad_lat,
                max(cat['latitude']) + pad_lat,
                min(cat['longitude']) - pad_lon,
                max(cat['longitude']) + pad_lon]

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
        s=np.array(cat["magnitude"]),
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
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]
