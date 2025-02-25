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

    It is possible to pass a reference point, which will be used as the
    origin of the local coordinate system. If no reference point is passed,
    the origin will be set to (0, 0, 0). The projection of the reference point
    is assumed to be the same as the local projection, unless specified
    otherwise in the ``ref_proj`` kwarg.

    Notes:
        4326 as well as eg 2056 are 2D coordinate systems, so altitude
        is not taken into account and only calculated in reference to the ref.

    """

    def __init__(
        self,
        local_proj: int | str,
        ref_easting: float = 0.0,
        ref_northing: float = 0.0,
        ref_altitude: float = 0.0,
        external_proj: int | str = 4326,
        ref_proj: int | str | None = None,
    ):
        """
        Constructor of CoordinateTransformer object.

        Args:
            local_proj:     Projection of local CRS,
                            eg. str('epsg:2056') or int(2056).
            ref_easting:    Reference easting.
            ref_northing:   Reference northing.
            ref_altitude:   Reference altitude.
            external_proj:  Projection of external CRS,
                            eg. str('epsg:4326') or int(4326).
            ref_proj:       Projection of reference coordinates,
                            defaults to 'local_proj'.
        """
        self.ref_easting = ref_easting
        self.ref_northing = ref_northing
        self.ref_altitude = ref_altitude
        self.local_proj = local_proj
        self.external_proj = external_proj

        if ref_proj is not None:
            tr = Transformer.from_proj(ref_proj, local_proj, always_xy=True)
            self.ref_easting, self.ref_northing = tr.transform(
                self.ref_easting, self.ref_northing
            )

        self.transformer_to_local = Transformer.from_proj(
            self.external_proj, self.local_proj, always_xy=True
        )
        self.transformer_to_external = Transformer.from_proj(
            self.local_proj, self.external_proj, always_xy=True
        )

    def to_local_coords(
        self,
        lon: float | list,
        lat: float | list,
        altitude: float | list = None,
    ):
        """
        Transform geographic coordinates to local coordinates.

        Args:
            lon:        Longitude of coordinate.
            lat:        Latitude of coordinate.
            altitude:   Altitude of coordinate.

        Returns:
            x:          Easting, northing and altitude in local CRS
                    relative to reference.

        Examples:
            >>> from seismostats.utils import CoordinateTransformer

            >>> ref_easting = 2642690
            >>> ref_northing = 1205590
            >>> ct = CoordinateTransformer(2056, ref_easting, ref_northing)
            >>> ct.to_local_coords(8.0, 47.0, 500)
            (5.414059637114406, 0.5106921272817999, 500)

        See also:
            :func:`~seismostats.utils.coordinates.from_local_coords`
        """
        enu = self.transformer_to_local.transform(lon, lat, altitude)
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
        altitude: float | list = None,
    ):
        """
        Transform local coordinates to geographic coordinates.

        Args:
            easting:    Easting of coordinate.
            northing:   Northing of coordinate.
            altitude:   Altitude of coordinate.

        Returns:
            x:          Longitude, latitude, altitude in local CRS
                    relative to reference.

        Examples:
            >>> from seismostats.utils import CoordinateTransformer

            >>> ref_easting = 2642690
            >>> ref_northing = 1205590
            >>> ct = CoordinateTransformer(2056, ref_easting, ref_northing)
            >>> ct.from_local_coords(0, 111000, 500)
            (8.010557515399034, 47.99827983628076, 500.0)
            >>> ct.from_local_coords(0, 0, 500)
            (7.999928777183829, 46.99999576464952, 500.0)


        See also:
            :func:`~seismostats.utils.coordinates.to_local_coords`
        """
        easting_0 = np.array(easting) + self.ref_easting
        northing_0 = np.array(northing) + self.ref_northing

        if altitude is not None:
            new_altitude = np.array(altitude) + self.ref_altitude
        else:
            new_altitude = None

        enu = self.transformer_to_external.transform(
            easting_0, northing_0, new_altitude
        )
        if new_altitude is None:
            enu = (enu[0], enu[1], None)

        return enu

    def polygon_to_local_coords(self, polygon: Polygon) -> Polygon:
        """
        Transform polygon from geographic coordinates to local coordinates.

        Args:
            polygon:    Shapely polygon.

        Returns:
            x:          Shapely polygon in local coordinates.

        Examples:
            >>> from seismostats.utils import CoordinateTransformer
            >>> from shapely.geometry import Polygon

            >>> ref_easting = 2642690
            >>> ref_northing = 1205590
            >>> ct = CoordinateTransformer(2056, ref_easting, ref_northing)
            >>> polygon = Polygon([(8.0, 47.0), (8.0, 48.0), (9.0, 48.0)])
            >>> local_polygon = ct.polygon_to_local_coords(polygon)
            >>> x, y = local_polygon.exterior.xy
            >>> x
            array('d', [5.4201549254357815, -789.3621860183775,
                73841.78791614715, 5.4201549254357815])
            >>> y
            array('d', [0.5222872239537537, 111185.62139117694,
                112195.20617892314, 0.5222872239537537])

        See also:
            :func:`~seismostats.utils.coordinates.polygon_from_local_coords`
        """
        new_polygon = transform(self.transformer_to_local.transform, polygon)
        translated_polygon = translate(
            new_polygon,
            xoff=-self.ref_easting,
            yoff=-self.ref_northing,
            zoff=-self.ref_altitude,
        )
        return translated_polygon

    def polygon_from_local_coords(self, polygon: Polygon) -> Polygon:
        """
        Transform polygon from local coordinates to geographic coordinates.

        Args:
            polygon: Shapely polygon.

        Returns:
            x: Shapely polygon in geographic coordinates.

        Examples:
            >>> from seismostats.utils import CoordinateTransformer
            >>> from shapely.geometry import Polygon

            >>> ref_easting = 2642690
            >>> ref_northing = 1205590
            >>> ct = CoordinateTransformer(2056, ref_easting, ref_northing)
            >>> polygon = Polygon([(0, 0), (0, 111000), (111000, 111000)])
            >>> external_polygon = ct.polygon_from_local_coords(polygon)
            >>> x, y = external_polygon.exterior.xy
            >>> x
            array('d', [7.999928695956813, 8.010557432492037,
                9.497252799712923, 7.999928695956813])
            >>> y
            array('d', [46.99999566074669, 47.99827972304315,
                47.98154598785103, 46.99999566074669])
        """
        translated_polygon = translate(
            polygon,
            xoff=self.ref_easting,
            yoff=self.ref_northing,
            zoff=self.ref_altitude,
        )
        new_polygon = transform(
            self.transformer_to_external.transform, translated_polygon
        )
        return new_polygon


def bounding_box_to_polygon(x_min, x_max, y_min, y_max, srid=None) -> Polygon:
    """
    Create a shapely Polygon from a bounding box.

    Args:
        x_min:  Minimum x coordinate.
        x_max:  Maximum x coordinate.
        y_min:  Minimum y coordinate.
        y_max:  Maximum y coordinate.
        srid:   Spatial reference system identifier.

    Returns:
        x: Shapely polygon in geographic coordinates.

    Examples:
        >>> from seismostats.utils import bounding_box_to_polygon
        >>> polygon = bounding_box_to_polygon(0, 1, 0, 1)
        >>> x,y = polygon.exterior.xy
        >>> x
        array('d', [1.0, 1.0, 0.0, 0.0, 1.0])
        >>> y
        array('d', [0.0, 1.0, 1.0, 0.0, 0.0])
    """
    bbox = (x_min, y_min, x_max, y_max)
    return geometry.box(*bbox, ccw=True)


def polygon_to_bounding_box(
    polygon: Polygon,
) -> tuple[float, float, float, float]:
    """
    Get the bounding box of a Polygon.

    Args:
        polygon: Shapely Polygon.

    Returns:
        tuple: The extend of the polygon with minimum and
            maximum coordinates of x and y.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        >>> polygon_to_bounding_box(polygon)
        (0.0, 0.0, 1.0, 1.0)
    """
    (minx, miny, maxx, maxy) = polygon.bounds
    return (minx, miny, maxx, maxy)
