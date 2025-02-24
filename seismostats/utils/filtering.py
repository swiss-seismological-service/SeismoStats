import pandas as pd
from shapely.geometry import Point, Polygon


def cat_intersect_polygon(
    cat: pd.DataFrame, polygon_vertices: list[tuple]
) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the rows with points inside a given
    polygon.

    Args:
        cat:                DataFrame with columns 'latitude' and 'longitude'
                        containing the points to be checked.
        polygon_vertices:   List of (x, y) tuples representing
                        the vertices of the polygon to be checked against.

    Returns:
        filtered_cat:       DataFrame containing only the rows with points
                        inside the polygon.

    """
    # Make a copy of the DataFrame
    cat_copy = cat.copy()
    # Add a new column to the DataFrame
    # indicating whether each point is inside the polygon
    cat_copy["inside_polygon"] = cat_copy.apply(
        lambda row: Polygon(polygon_vertices).intersects(
            Point(row["latitude"], row["longitude"])
        ),
        axis=1,
    )

    # Filter the DataFrame to only include rows where
    # the point is inside the polygon
    filtered_cat = (
        cat_copy.query("inside_polygon")
        .drop("inside_polygon", axis=True)
        .copy()
    )

    return filtered_cat
