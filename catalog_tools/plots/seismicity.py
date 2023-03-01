import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from cartopy.mpl import gridliner
from cartopy.io import shapereader
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


from typing import Optional


def plot_in_space(
        cat: pd.DataFrame,
        country: Optional[str] = None,
        resolution: str = '10m',
        category: str = 'cultural',
        name: str = 'admin_0_countries'
):
    # request data for use by geopandas
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)

    # stamen_terrain = cimgt.Stamen('terrain-background', desired_tile_form="L")
    stamen_terrain = cimgt.Stamen('terrain-background')

    # projections that involved
    st_proj = stamen_terrain.crs  # projection used by Stamen images
    ll_proj = ccrs.PlateCarree()  # CRS for raw long/lat

    ax = plt.subplot(projection=st_proj)

    if country is not None:
        # get geometry of a country
        poly = [df.loc[df['ADMIN'] == country]['geometry'].values[0]]
        pad_lat = abs(poly[0].bounds[0] - poly[0].bounds[2]) * 0.05
        pad_lon = abs(poly[0].bounds[1] - poly[0].bounds[3]) * 0.05
        exts = [poly[0].bounds[0] - pad_lat, poly[0].bounds[2] + pad_lat,
                poly[0].bounds[1] - pad_lon, poly[0].bounds[3] + pad_lon]
    else:
        cat['latitude']
        pad_lat = abs(max(cat['latitude']) - min(cat['latitude'])) * 0.05
        pad_lon = abs(max(cat['longitude']) - min(cat['longitude'])) * 0.05
        exts = [min(cat['latitude']) - pad_lat,
                max(cat['latitude']) + pad_lat,
                min(cat['longitude']) - pad_lon,
                max(cat['longitude']) + pad_lon]

    ax.set_extent(exts, crs=ll_proj)
    ax.add_image(stamen_terrain, 8, alpha=0.6)
    # ax.add_image(stamen_terrain, 8, cmap='Greys_r', alpha=0.8)

    if country is not None:
        msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
        msk_stm = st_proj.project_geometry(msk, ll_proj)
        ax.add_geometries(msk_stm, st_proj, facecolor='white', edgecolor='grey',
                          alpha=0.6)

    # gridlines
    gl = ax.gridlines(crs=ll_proj, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.bottom_labels = False
    gl.right_labels = False

    ax.scatter(
        cat["latitude"],
        cat["longitude"],
        c='blue',
        edgecolor='k',
        s=np.array(cat["magnitude"]),
        zorder=100,
        transform=ccrs.PlateCarree(),
        linewidth=0.5, alpha=0.8,
        label='observed earthquakes'
    )
    ax.legend(fancybox=False).set_zorder(20)


def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]