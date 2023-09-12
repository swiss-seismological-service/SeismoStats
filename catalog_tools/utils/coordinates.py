import numpy as np
from pyproj import Transformer


class CoordinateTransformer:
    """
    Class to transform between a external geographic (default ESPG:4326,
    also known as WGS84), and a local cartesian CRS.

    Any EPSG code or proj4 string can be used for the local_proj input,
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

        :param local_proj: int (epsg) or string (proj) of local CRS.
        :param ref_easting: reference easting for local coordinates.
        :param ref_northing: reference northing for local coordinates.
        :param ref_altitude: reference altitude for local coordinates.
        :param external_proj: int or string of geographic coordinates.
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

        :param lon: longitude
        :param lat: latitude
        :param altitude: altitude
        :returns: Easting, northing and altitude in local CRS relative to ref.
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

        :param easting: easting
        :param northing: northing
        :param altitude: altitude
        :returns: longitude, latitude, altitude in local CRS relative to ref.
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
