import pandas as pd
from shapely.geometry import Polygon, Point


def df_intersect_polygon(df, polygon_vertices):
    """Returns a DataFrame containing
    only the rows with points inside a given polygon.

    Args:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'latitude' and 'longitude'
        containing the points to be checked.
    polygon_vertices : list of tuples
        List of (x, y) tuples representing
        the vertices of the polygon to be checked against.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the rows with points inside the polygon.

    """
    # Make a copy of the DataFrame
    df_copy = df.copy()
    # Add a new column to the DataFrame
    # indicating whether each point is inside the polygon
    df_copy['inside_polygon'] = df_copy.apply(
        lambda row: Polygon(polygon_vertices).intersects(
            Point(row['latitude'], row['longitude'])), axis=1)

    # Filter the DataFrame to only include rows where
    # the point is inside the polygon
    filtered_df = df_copy.query('inside_polygon').drop(
        'inside_polygon', axis=True).copy()

    return filtered_df
