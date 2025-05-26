import numpy as np
import pytest

from seismostats.utils.binning import (bin_to_precision, get_cum_fmd, get_fmd,
                                       normal_round, _normal_round_to_int,
                                       binning_test)


@pytest.mark.parametrize(
    "x, rounded_value",
    [(0.235, 0), (-0.235, 0), (-0.5, -1),
     (4.499, 4), (4.5, 5), (5.5, 6), (6.5, 7)]
)
def test_normal_round_to_int(x: float, rounded_value: int):
    y = _normal_round_to_int(x)
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


def test_bin_to_precision_none():
    with pytest.raises(ValueError):
        bin_to_precision(None, 0.1)
    with pytest.raises(ValueError):
        bin_to_precision([1, 2, 3], 0)
    with pytest.raises(TypeError):
        bin_to_precision([0.23, 0.56, 0.78])


@pytest.mark.parametrize(
    "magnitudes, delta_m, bins, c_counts, bin_position",
    [(np.array([0.20990507, 0.04077336, 0.27906596, 0.57406287, 0.64256544,
                0.07293118, 0.58589873, 0.02678655, 0.27631233, 0.17682814]),
      0.1, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      np.array([10, 8, 7, 5, 3, 3, 3]), 'center'),
     (np.array([0.02637757, 0.06353823, 0.10257919, 0.54494906, 0.03928375,
                0.08825028, 0.77713586, 0.54553981, 0.69315583, 0.06656642,
                0.29035447, 0.2051877, 0.30858087, 0.68896342, 0.03328782,
                0.45016109, 0.40779409, 0.06788892, 0.02684032, 0.56140282,
                0.29443359, 0.36328762, 0.17124489, 0.02154936, 0.36461541,
                0.03613088, 0.15798366, 0.09111875, 0.16169287, 0.11986668,
                0.10232035, 0.72695761, 0.19484174, 0.0459675, 0.40514163,
                0.08979514, 0.0442659, 0.18672424, 0.21239088, 0.02287468,
                0.1244267, 0.04939361, 0.11232758, 0.02706083, 0.04275401,
                0.02732529, 0.83884229, 0.4147758, 0.07416183, 0.05636252]),
      0.2, np.array([-0.1, 0.1, 0.3, 0.5, 0.7]), np.array([50, 29, 15, 8, 3]),
      'left')]
)
def test_get_cum_fmd(magnitudes: np.ndarray, delta_m: float,
                     bins: np.ndarray, c_counts: np.ndarray, bin_position: str):
    errors = []
    nbins, nc_counts, nmags = get_cum_fmd(
        magnitudes, delta_m, bin_position=bin_position)
    if not np.allclose(bins, nbins, atol=1e-10):
        errors.append("Incorrect bin values.")
    if not np.allclose(c_counts, nc_counts, atol=1e-10):
        errors.append("Incorrect cumulative counts.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize(
    "magnitudes, fmd_bin, bins, counts, bin_position",
    [(np.array([0.20990507, 0.04077336, 0.27906596, 0.57406287, 0.64256544,
                0.07293118, 0.58589873, 0.02678655, 0.27631233, 0.17682814]),
      0.1, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
      np.array([2, 1, 2, 2, 0, 0, 3]), 'center'),
     (np.array([0.02637757, 0.06353823, 0.10257919, 0.54494906, 0.03928375,
                0.08825028, 0.77713586, 0.54553981, 0.69315583, 0.06656642,
                0.29035447, 0.2051877, 0.30858087, 0.68896342, 0.03328782,
                0.45016109, 0.40779409, 0.06788892, 0.02684032, 0.56140282,
                0.29443359, 0.36328762, 0.17124489, 0.02154936, 0.36461541,
                0.03613088, 0.15798366, 0.09111875, 0.16169287, 0.11986668,
                0.10232035, 0.72695761, 0.19484174, 0.0459675, 0.40514163,
                0.08979514, 0.0442659, 0.18672424, 0.21239088, 0.02287468,
                0.1244267, 0.04939361, 0.11232758, 0.02706083, 0.04275401,
                0.02732529, 0.83884229, 0.4147758, 0.07416183, 0.05636252]),
      0.2, np.array([-0.1, 0.1, 0.3, 0.5, 0.7]), np.array([21, 14, 7, 5, 3]),
      'left')]
)
def test_get_fmd(magnitudes: np.ndarray, fmd_bin: float,
                 bins: np.ndarray, counts: np.ndarray, bin_position):
    errors = []
    nbins, ncounts, nmags = get_fmd(
        magnitudes, fmd_bin, bin_position=bin_position)

    if not np.allclose(bins, nbins, atol=1e-10):
        errors.append("Incorrect bin values.")
    if not np.allclose(counts, ncounts, atol=1e-10):
        errors.append("Incorrect counts.")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

    with pytest.raises(ValueError):
        get_fmd(magnitudes, 0, bin_position=bin_position)


def test_test_binning():
    a = [0.2, 0.4, 0.6, 0.8, 1.0]
    assert binning_test(a, 0.1)
    assert binning_test(a, 0.2)
    assert not binning_test(a, 0.02)

    a = [1, 4, 7, 10, 1.3]
    assert binning_test(a, 0.1)
