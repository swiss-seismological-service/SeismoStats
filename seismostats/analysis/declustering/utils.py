import numpy as np
import pandas as pd


def decimal_year(catalogue: pd.DataFrame) -> np.ndarray:
    """
    Converts the column 'time' of the catalogue to decimal years,
    using a precision of seconds.

    Args:
        catalogue: A pandas DataFrame with a column 'time' in datetime format.
    Note:
        Used for comparing time windows from BaseDistanceTimeWindow
    Returns:
        A numpy array with the decimal years for each event in the catalogue.
        Using a precision of seconds.
    """
    time = catalogue["time"].astype("datetime64[s]", errors="ignore")
    day = time.dt.dayofyear + time.dt.second / 86400
    return (time.dt.year + day / 365).values


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
