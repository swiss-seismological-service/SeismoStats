from catalog_tools.binning import normal_round_to_int, normal_round
from catalog_tools.binning import bin_to_precision, bin_magnitudes

import pytest


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
    "x, delta_m, rounded_value",
    [(0.235, 0.1, 0.2), (-0.235, 0.2, -0.2), 
     (4.499, 0.01, 4.5), (4.5, 0.2, 4.6)]
)
def test_bin_to_precision(x: float, delta_m: float, rounded_value: float):
    y = bin_to_precision(x, delta_m)
    assert y == rounded_value


@pytest.mark.parametrize(
    "mags, delta_m, rounded_list",
    [
        ([0.235, -0.235, 4.499, 4.5, 6, 0.1, 1.6],
         0.1,
         [0.2, -0.2, 4.5, 4.5, 6, 0.1, 1.6]),
        ([0.235, -0.235, 4.499, 5.5, 6, 0.1, 1.6],
         0.2,
         [0.2, -0.2, 4.4, 5.6, 6, 0.2, 1.6])
    ]
)
def test_bin_magnitudes(mags: list, delta_m: float, rounded_list: list):
    y = bin_magnitudes(mags, delta_m)
    assert y == rounded_list
