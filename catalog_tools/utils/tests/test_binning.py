from catalog_tools.utils.binning import normal_round_to_int, normal_round
from catalog_tools.utils.binning import bin_to_precision

import pytest
import numpy as np


@pytest.mark.parametrize(
    "x, rounded_value",
    [(0.235, 0), (-0.235, 0), (-0.5, -1),
     (4.499, 4), (4.5, 5), (5.5, 6), (6.5, 7)]
)
def test_normal_round_to_int(x: float, rounded_value: int):
    y = normal_round_to_int(x)
    assert y == rounded_value


@pytest.mark.parametrize(
    "x, n, rounded_value",
    [(0.235, 2, 0.24), (-0.235, 2, -0.24), (4.499, 2, 4.5), (4.5, 0, 5)]
)
def test_normal_round(x: float, n: int, rounded_value: float):
    y = normal_round(x, n)
    assert y == rounded_value


@pytest.mark.parametrize(
    "x, delta_x, rounded_value",
    [
        (np.array([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6]),
         0.1,
         np.array([0.2, -0.2, 4.5, 4.5, 6, 0.1, 1.6])),
        (np.array([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6]),
         0.2,
         np.array([0.2, -0.2, 4.4, 5.6, 6, 0.2, 1.6])),
        ([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6],
         0.2,
         [0.2, -0.2, 4.4, 5.6, 6, 0.2, 1.6])
    ]
)
def test_bin_to_precision(x: np.ndarray, delta_x: float,
                          rounded_value: np.ndarray):
    y = bin_to_precision(x, delta_x)
    assert (y == rounded_value).all()
