
import numpy as np
import pytest

from seismostats.analysis.bvalue.weichert import estimate_b_weichert
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned


def _create_test_catalog_poisson(a_val_true: float, b_val_true: float):
    """
    creates a synthetic catalog with magnitudes
    between 4 and 7.9 with unequal completeness periods. To be used
    for testing relevant recurrence parameter estimators.
    """

    # assume a catalog from year 1000 to end of 1999
    # with completeness as follows:
    completeness_table = np.array(
        [[3.95, 1940], [4.95, 1880], [5.95, 1500], [6.95, 1000]]
    )

    end_year = 2000
    mmax = 7.95

    obs_mags = []
    obs_times = []
    for ii in range(len(completeness_table)):
        bin_lower_edge, cyear_lower = completeness_table[ii]
        if ii == len(completeness_table) - 1:
            bin_upper_edge = mmax
        else:
            bin_upper_edge, _ = completeness_table[ii + 1]

        # get expected annual rates in completeness bin
        exp_rate = 10 ** (a_val_true - b_val_true * bin_lower_edge) - 10 ** (
            a_val_true - b_val_true * bin_upper_edge
        )

        # sample observed earthquakes over 1,000 year period
        obs_countsi = np.random.poisson(exp_rate * (end_year - cyear_lower))
        obs_mags.append(
            simulate_magnitudes_binned(
                n=obs_countsi,
                b=1,
                mc=bin_lower_edge + 0.05,
                delta_m=0.1,
                mag_max=bin_upper_edge,
            )
        )
        obs_yearsi = np.random.randint(cyear_lower, end_year, obs_countsi)
        obs_times.append(
            np.array(["%d-06-15" % i for i in obs_yearsi], dtype="datetime64")
        )

    # add some earthquakes in incomplete years
    mags_inc = np.concatenate(
        [
            np.random.randint(40, 50, 100) / 10,
            np.random.randint(50, 60, 10) / 10,
            np.random.randint(60, 70, 1) / 10,
        ]
    )
    years_inc = np.concatenate(
        [
            np.random.randint(1000, 1940, 100),
            np.random.randint(1000, 1880, 10),
            np.random.randint(1000, 1500, 1),
        ]
    )
    times_inc = np.array(
        ["%d-06-15" % i for i in years_inc], dtype="datetime64"
    )

    # merge complete and incomplete earthquakes
    mags = np.concatenate([*obs_mags, mags_inc])
    times = np.concatenate([*obs_times, times_inc])
    return mags, times


@pytest.mark.parametrize("a_val_true,b_val_true,precision", [(7, 1, 0.01)])
def test_estimate_b_weichert(
    a_val_true: float, b_val_true: float, precision: float
):
    mags, times = _create_test_catalog_poisson(a_val_true, b_val_true)

    (
        b_val,
        std_b_val,
        rate_at_mref,
        std_rate_at_mref,
        a_val,
    ) = estimate_b_weichert(
        magnitudes=mags,
        times=times,
        completeness_table=np.array(
            [[3.95, 1940], [4.95, 1880], [5.95, 1500], [6.95, 1000]]
        ),
        mag_max=7.95,
        last_year=2000,
        delta_m=0.1,
        b_parameter="b_value",
    )

    assert abs(b_val_true - b_val) / b_val_true <= precision
    assert abs(a_val_true - a_val) / a_val_true <= precision
