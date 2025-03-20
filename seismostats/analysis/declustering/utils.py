import numpy as np


def haversine(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    target_longitude: float,
    target_latitude: float,
    earth_rad=6371.227,
) -> np.ndarray:
    """
    The haversine formula determines the great-circle distance between
    two points on a sphere given their longitudes and latitudes.
    Returns the distance in km between each pair of locations given in
    longitudes and latitudes to (target_longitude, target_latitude).

    Source:
        https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        longitudes:         Array of longitudes in degrees.
        latitudes:          Array of latitudes in degrees.
        target_longitude:   Longitude in degrees.
        target_latitude:    Latitude in degrees.
        earth_rad:          Radius of the Earth in km.

    Returns:
        Array of distances in km.

    Examples:
        >>> import numpy as np
        >>> from seismostats.analysis.declustering.utils import haversine
        >>> longitudes = np.array([8.0, 8.0])
        >>> latitudes = np.array([47.0, 48.0])
        >>> target_longitude = 8.0
        >>> target_latitude = 47.0
        >>> haversine(longitudes, latitudes, target_longitude, target_latitude)
        array([0.   , 111.19888854])
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
        np.cos(latitudes)
        * np.cos(target_latitude)
        * (np.sin(dlon / 2.0) ** 2.0)
    )
    distance = (
        2.0 * earth_rad * np.arctan2(np.sqrt(aval), np.sqrt(1 - aval))
    ).T
    return distance
