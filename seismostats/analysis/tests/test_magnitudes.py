from numpy.testing import assert_almost_equal, assert_equal

from seismostats.analysis.magnitudes import apply_edwards


def test_apply_edwards():
    mag_types = ['MLh', 'MLhc', 'Ml', 'Mw']
    mags = [2.5, 3.5, 4.5, 5.5]

    result = [apply_edwards(typ, mag).tolist()
              for (typ, mag) in zip(mag_types, mags)]

    verify = [['Mw_converted', 2.506875], ['Mw_converted', 3.2734749999999995],
              ['Mw_converted', 4.138274999999999], ['Mw', 5.5]]

    for i, res in enumerate(result):
        assert_almost_equal(res[1], verify[i][1])
        assert_equal(res[0], verify[i][0])
