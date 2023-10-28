import numpy as np
import pandas as pd

# import functions to be tested
from seismostats.utils.filtering import cat_intersect_polygon


def test_cat_intersect_polygon():
    df_test = pd.DataFrame({
        'latitude': [40.0, 40.1, 39.9],
        'longitude': [-105.0, -105.1, -104.9],
        'magnitude': [2.3, 4.1, 1.8]
    })

    # Define a test polygon as a list of (x, y) tuples
    polygon_vertices = np.array(
        [[39.8, -105.2], [40.2, -105.2], [40.2, -104.95], [39.8, -104.95]])

    # Call the df_intersect_polygon function
    filtered_df = cat_intersect_polygon(df_test, polygon_vertices)

    # Define the expected output DataFrame
    expected_df = pd.DataFrame({
        'latitude': [40.0, 40.1],
        'longitude': [-105.0, -105.1],
        'magnitude': [2.3, 4.1]
    })

    # Test that the output DataFrame is correct
    assert filtered_df.equals(expected_df), 'not the same df'

    # Test the function with a more complex polygon
    complex_polygon_vertices = np.array(
        [[0, 0], [0, 5], [5, 5], [5, 0], [2.5, 2.5]])
    df = pd.DataFrame({
        'latitude': [1, 2.1, 3, 4, 2.5],
        'longitude': [1, 2, 3, 4, 2.5]
    })
    filtered_df = cat_intersect_polygon(df, complex_polygon_vertices)
    expected_df = pd.DataFrame({
        'latitude': [1, 3, 4, 2.5],
        'longitude': [1, 3, 4, 2.5]
    }, index=[0, 2, 3, 4])
    assert filtered_df.equals(expected_df), 'not the same df'
