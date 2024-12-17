

import unittest

from seismostats.utils.coordinates import CoordinateTransformer

WGS84_PROJ = "epsg:4326"
SWISS_PROJ = "epsg:2056"

# Coordinates roughly centred on Zurich
LON = 8.54
LAT = 47.3

REF_EASTING = 2683295.5134568703
REF_NORTHING = 1239374.8568319362
REF_ALTITUDE = 0.0

REF_LON = 8.54
REF_LAT = 47.3

# Coordinates for Bedretto lab
BEDRETTO_LAB_ORIGIN_EASTING = 2679720.70
BEDRETTO_LAB_ORIGIN_NORTHING = 1151600.13
BEDRETTO_LAB_ORIGIN_ALTITUDE = 1485.00

# Coordinates of 'injection point' in both coordinate systems
MOCK_INJ_EASTING = 100.0
MOCK_INJ_NORTHING = 200.0
MOCK_INJ_ALTITUDE_LOCAL = -300.0

MOCK_INJ_LON = 8.478714065123917
MOCK_INJ_LAT = 46.51275035512119
MOCK_INJ_ALTITUDE = 1185.0


class TransformationCoordsTestCase(unittest.TestCase):
    """
    Test case for checking back and forth transformation
    between source proj and local coordinate system.
    """

    def test_simple_conversion(self):
        """
        Test transformer with no reference point, so that the local coordinate
        system is the same as the swiss grid.
        """
        transformer = CoordinateTransformer(SWISS_PROJ,
                                            external_proj=WGS84_PROJ)
        easting, northing, altitude = transformer.\
            to_local_coords(
                LON, LAT)
        self.assertAlmostEqual(easting, REF_EASTING)
        self.assertAlmostEqual(northing, REF_NORTHING)

        lon, lat, _ = transformer.from_local_coords(
            easting, northing)
        self.assertAlmostEqual(lat, LAT)
        self.assertAlmostEqual(lon, LON)

    def test_reference_point(self):
        """
        Test transformer with a reference point, so that the local coordinate
        system is centred on Zurich. Therefore the easting and northing
        calculated should be 0,0 as the reference point is equal to the
        centre of the local coordinate system.
        """
        transformer = CoordinateTransformer(
            SWISS_PROJ, external_proj=WGS84_PROJ,
            ref_easting=REF_EASTING, ref_northing=REF_NORTHING)
        easting, northing, altitude = transformer.\
            to_local_coords(
                LON, LAT)
        self.assertAlmostEqual(easting, 0.0)
        self.assertAlmostEqual(northing, 0.0)

        lon, lat, _ = transformer.from_local_coords(
            easting, northing)
        self.assertAlmostEqual(lat, LAT)
        self.assertAlmostEqual(lon, LON)

    def test_bedretto_transformation(self):
        """
        Test transformer with the reference point at the Bedretto
        lab. Convert local coordinates into wgs84.
        """
        transformer = CoordinateTransformer(
            SWISS_PROJ, external_proj=WGS84_PROJ,
            ref_easting=BEDRETTO_LAB_ORIGIN_EASTING,
            ref_northing=BEDRETTO_LAB_ORIGIN_NORTHING,
            ref_altitude=BEDRETTO_LAB_ORIGIN_ALTITUDE)

        lon, lat, altitude = transformer.from_local_coords(
            MOCK_INJ_EASTING,
            MOCK_INJ_NORTHING,
            altitude=MOCK_INJ_ALTITUDE_LOCAL)

        self.assertAlmostEqual(lat, MOCK_INJ_LAT)
        self.assertAlmostEqual(lon, MOCK_INJ_LON)
        self.assertAlmostEqual(altitude, MOCK_INJ_ALTITUDE)

    def test_reference_proj(self):
        """
        Test transformer with a reference point and a different
        reference projection. The reference projection is set to
        WGS84, so the reference point is transformed to the local
        coordinate system.
        """
        transformer = CoordinateTransformer(
            SWISS_PROJ, ref_easting=REF_LON, ref_northing=REF_LAT,
            ref_proj=WGS84_PROJ)

        easting, northing, altitude = transformer.\
            to_local_coords(
                LON, LAT)
        self.assertAlmostEqual(easting, 0.0)
        self.assertAlmostEqual(northing, 0.0)

        lon, lat, _ = transformer.from_local_coords(
            easting, northing)
        self.assertAlmostEqual(lat, LAT)
        self.assertAlmostEqual(lon, LON)
