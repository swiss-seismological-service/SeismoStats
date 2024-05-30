import numpy as np
from pyproj import Transformer
from shapely import geometry
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import transform


class CoordinateTransformer:
    """
    Class to transform between a external geographic (default ESPG:4326,
    also known as WGS84), and a local cartesian CRS.

    Any EPSG code or proj4 string can be used for the ``local_proj`` input,
    for instance 2056 to represent the swiss coordinate system, or
    "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    to represent a UTM coordinate system.

    Caveat: 4326 as well as eg 2056 are 2D coordinate systems, so altitude
    is not taken into account and only calculated in reference to the ref.

    """

    def __init__(
            self,
            local_proj: int | str,
            ref_easting: float = 0.0,
            ref_northing: float = 0.0,
            ref_altitude: float = 0.0,
            external_proj: int | str = 4326):
        """
        Constructor of CoordinateTransformer object.
        Args:
            local_proj: int (epsg) or string (proj) of local CRS.
            ref_easting: reference easting for local coordinates.
            ref_northing: reference northing for local coordinates.
            ref_altitude: reference altitude for local coordinates.
            external_proj: int or string of geographic coordinates.
        """
        self.ref_easting = ref_easting
        self.ref_northing = ref_northing
        self.ref_altitude = ref_altitude
        self.local_proj = local_proj
        self.external_proj = external_proj

        self.transformer_to_local = Transformer.from_proj(
            self.external_proj, self.local_proj, always_xy=True)
        self.transformer_to_external = Transformer.from_proj(
            self.local_proj, self.external_proj, always_xy=True)

    def to_local_coords(self,
                        lon: float | list,
                        lat: float | list,
                        altitude: float | list = None):
        """
        Transform geographic coordinates to local coordinates.
        Args:
            lon: longitude
            lat: latitude
            altitude: altitude

        Returns:
            Easting, northing and altitude in local CRS relative to ref.
        """
        enu = \
            self.transformer_to_local.transform(lon, lat, altitude)
        easting = np.array(enu[0]) - self.ref_easting
        northing = np.array(enu[1]) - self.ref_northing

        if altitude is not None:
            new_altitude = np.array(enu[2]) - self.ref_altitude
        else:
            new_altitude = None

        return easting, northing, new_altitude

    def from_local_coords(
            self,
            easting: float | list,
            northing: float | list,
            altitude: float | list = None):
        """
        Transform local coordinates to geographic coordinates.

        Args:
            easting: easting
            northing: northing
            altitude: altitude

        Returns:
            longitude, latitude, altitude in local CRS relative to ref.
        """
        easting_0 = np.array(easting) + self.ref_easting
        northing_0 = np.array(northing) + self.ref_northing

        if altitude is not None:
            new_altitude = np.array(altitude) + self.ref_altitude
        else:
            new_altitude = None

        enu = self.transformer_to_external.transform(easting_0,
                                                     northing_0,
                                                     new_altitude)
        if new_altitude is None:
            enu = (enu[0], enu[1], None)

        return enu

    def polygon_to_local_coords(self, polygon: Polygon) -> Polygon:
        """
        Transform polygon from geographic coordinates to local coordinates.

        Args:
            polygon: shapely polygon

        Returns:
            shapely polygon in local coordinates
        """
        new_polygon = transform(
            self.transformer_to_local.transform, polygon)
        translated_polygon = translate(
            new_polygon, xoff=-self.ref_easting,
            yoff=-self.ref_northing, zoff=-self.ref_altitude)
        return translated_polygon

    def polygon_from_local_coords(self, polygon: Polygon) -> Polygon:
        """
        Transform polygon from local coordinates to geographic coordinates.

        Args:
            polygon: shapely polygon

        Returns:
            shapely polygon in geographic coordinates
        """
        translated_polygon = translate(
            polygon, xoff=self.ref_easting,
            yoff=self.ref_northing, zoff=self.ref_altitude)
        new_polygon = transform(
            self.transformer_to_external.transform, translated_polygon)
        return new_polygon


def bounding_box_to_polygon(x_min, x_max, y_min, y_max, srid=None) -> Polygon:
    """
    Create a shapely Polygon from a bounding box.

    Args:
        x_min: minimum x coordinate
        x_max: maximum x coordinate
        y_min: minimum y coordinate
        y_max: maximum y coordinate
        srid: spatial reference system identifier
    """
    bbox = (x_min, y_min,
            x_max, y_max)
    return geometry.box(*bbox, ccw=True)


def polygon_to_bounding_box(polygon: Polygon) -> \
        tuple[float, float, float, float]:
    """
    Get the bounding box of a Polygon.

    Args:
        polygon: shapely Polygon

    Returns:
        tuple: The corner coordinates of the Polygon
    """
    (minx, miny, maxx, maxy) = polygon.bounds
    return (minx, miny, maxx, maxy)
