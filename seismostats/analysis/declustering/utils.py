import numpy as np


def haversine(longitudes: np.ndarray, latitudes: np.ndarray,
              target_longitude: float, target_latitude: float,
              earth_rad=6371.227) -> np.ndarray:
    """
    The haversine formula determines the great-circle distance
    between two points on a sphere given their longitudes and latitudes.
    Returns the distance in km between each pair of locations
    given in longitudes and latitudes to (target_longitude, target_latitude).

    Source:
        https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        longitudes: array of longitudes in degrees
        latitudes: array of latitudes in degrees
        target_longitude: longitude in degrees
        target_latitude: latitude in degrees
        earth_rad: radius of the Earth in km

    Returns:
        array of distances in km
    """
    cfact = np.pi / 180.0
    longitudes = cfact * longitudes
    latitudes = cfact * latitudes
    target_longitude = cfact * target_longitude
    target_latitude = cfact * target_latitude

    n = np.max(np.shape(longitudes))
    distance = np.zeros(n)

    # Perform distance calculation
    dlat = latitudes - target_latitude
    dlon = longitudes - target_longitude
    aval = (np.sin(dlat / 2.0) ** 2.0) + (
        np.cos(latitudes) * np.cos(target_latitude)
        * (np.sin(dlon / 2.0) ** 2.0)
    )
    distance = (
        2.0 * earth_rad * np.arctan2(np.sqrt(aval), np.sqrt(1 - aval))
    ).T
    return distance
