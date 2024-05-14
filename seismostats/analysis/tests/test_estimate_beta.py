import pickle
import warnings

import numpy as np
import pytest

# import functions to be tested
from seismostats.analysis.estimate_beta import (differences, estimate_b,
                                                estimate_b_laplace,
                                                estimate_b_positive,
                                                estimate_b_tinti,
                                                estimate_b_utsu,
                                                estimate_b_weichert,
                                                shi_bolt_confidence)
from seismostats.utils.binning import bin_to_precision
# import functions from other modules
from seismostats.utils.simulate_distributions import simulate_magnitudes


def simulate_magnitudes_w_offset(
    n: int, beta: float, mc: float, delta_m: float, mag_max: float = None
) -> np.ndarray:
    """This function simulates the magnitudes with the correct offset"""
    mags = simulate_magnitudes(n, beta, mc - delta_m / 2, mag_max)
    if delta_m > 0:
        mags = bin_to_precision(mags, delta_m)
    return mags


@pytest.mark.parametrize(
    "method, return_std, return_n, b_parameter",
    [
        ("tinti", True, True, "beta"),
        ("tinti", False, False, "b_value"),
        ("utsu", True, False, "beta"),
        ("utsu", False, True, "b_value"),
        ("positive", True, True, "beta"),
        ("positive", False, False, "b_value"),
        ("positive", True, False, "beta"),
        ("positive", False, True, "b_value"),
        ("laplace", True, True, "beta"),
        ("laplace", False, False, "b_value"),
    ],
)
def test_estimate_b(
    method: str,
    return_std: bool,
    return_n: bool,
    b_parameter: str,
):
    """this test only checks if the number of output is correct. the actual
    values are tested in the individual tests for each method"""
    mags = simulate_magnitudes_w_offset(
        n=100, beta=np.log(10), mc=0, delta_m=0.1
    )
    out = estimate_b(
        mags,
        mc=0,
        delta_m=0.1,
        b_parameter=b_parameter,
        method=method,
        return_std=return_std,
        return_n=return_n,
    )

    # assert that the correct number of values are returned
    assert np.size(out) == 1 + return_std + (
        return_n
        * (1 - (method == "utsu"))
        * (1 - (method == "tinti"))
        * (1 - (method == "laplace"))
    )

    # test that uncorrect binninng leads to error
    with pytest.raises(AssertionError) as excinfo:
        estimate_b(
            mags,
            mc=0,
            delta_m=0.2,
            b_parameter=b_parameter,
            method=method,
            return_std=return_std,
            return_n=return_n,
        )
    assert str(excinfo.value) == "magnitudes are not binned correctly"

    # test that magnitudes smaller than mc lkead to error
    with pytest.raises(AssertionError) as excinfo:
        estimate_b(
            mags,
            mc=1,
            delta_m=0.1,
            b_parameter=b_parameter,
            method=method,
            return_std=return_std,
            return_n=return_n,
        )
    assert str(excinfo.value) == "magnitudes below mc are present in the data"

    # test that warning is raised if smallest magnitude is much larger than mc
    with warnings.catch_warnings(record=True) as w:
        estimate_b(
            mags,
            mc=-1,
            delta_m=0.1,
            b_parameter=b_parameter,
            method=method,
            return_std=return_std,
            return_n=return_n,
        )
        assert w[-1].category == UserWarning


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [
        (1000000, 1.2 * np.log(10), 3, 0, "beta", 0.005),
        (1000000, np.log(10), 3, 0.1, "beta", 0.01),
    ],
)
def test_estimate_b_tinti(
    n: int,
    b: float,
    mc: float,
    delta_m: float,
    b_parameter: str,
    precision: float,
):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_tinti(mags, mc, delta_m, b_parameter=b_parameter)

    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,b_parameter,precision",
    [
        (1000000, 1.2 * np.log(10), 3, 0, "beta", 0.005),
        (1000000, np.log(10), 3, 0.1, "b_value", 0.01),
    ],
)
def test_estimate_b_utsu(
    n: int,
    beta: float,
    mc: float,
    delta_m: float,
    b_parameter: str,
    precision: float,
):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)

    if b_parameter == "b_value":
        b = beta / np.log(10)
    else:
        b = beta

    b_estimate = estimate_b_utsu(mags, mc, delta_m, b_parameter=b_parameter)
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "magnitudes,mag_diffs",
    [(np.array([1, -2, 3]), np.array([-3, 5, 2]))],
)
def test_differences(magnitudes: np.ndarray, mag_diffs: np.ndarray):
    y = differences(magnitudes)
    assert (y == mag_diffs).all()


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [
        (1000000, 1.2 * np.log(10), 3, 0, "beta", 0.005),
        (1000000, np.log(10), 3, 0.1, "beta", 0.01),
    ],
)
def test_estimate_b_positive(
    n: int,
    b: float,
    mc: float,
    delta_m: float,
    b_parameter: str,
    precision: float,
):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_positive(
        mags, delta_m=delta_m, b_parameter=b_parameter
    )
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [
        (1000, 1.2 * np.log(10), 3, 0, "beta", 0.15),
        (1000, np.log(10), 3, 0.1, "beta", 0.2),
    ],
)
def test_estimate_b_laplace(
    n: int,
    b: float,
    mc: float,
    delta_m: float,
    b_parameter: str,
    precision: float,
):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_laplace(
        mags, delta_m=delta_m, b_parameter=b_parameter
    )
    assert abs(b - b_estimate) / b <= precision


def _create_test_catalog_poisson(
        a_val_true: float,
        b_val_true: float):
    """
    creates a synthetic catalog with magnitudes
    between 4 and 7.9 with unequal completeness periods. To be used
    for testing relevant recurrence parameter estimators.
    """

    # assume a catalog from year 1000 to end of 1999
    # with completeness as follows:
    completeness_table = np.array([[3.95, 1940],
                                   [4.95, 1880],
                                   [5.95, 1500],
                                   [6.95, 1000]])

    end_year = 2000
    mmax = 7.95

    obs_mags = []
    obs_dates = []
    for ii in range(len(completeness_table)):
        bin_lower_edge, cyear_lower = completeness_table[ii]
        if ii == len(completeness_table) - 1:
            bin_upper_edge = mmax
        else:
            bin_upper_edge, _ = completeness_table[ii + 1]

        # get expected annual rates in completeness bin
        exp_rate = 10 ** (a_val_true - b_val_true
                          * bin_lower_edge) \
            - 10 ** (a_val_true - b_val_true
                     * bin_upper_edge)

        # sample observed earthquakes over 1,000 year period
        obs_countsi = np.random.poisson(exp_rate * (end_year - cyear_lower))
        obs_mags.append(simulate_magnitudes_w_offset(
            n=obs_countsi, beta=np.log(10), mc=bin_lower_edge + 0.05,
            delta_m=0.1,
            mag_max=bin_upper_edge))
        obs_yearsi = np.random.randint(cyear_lower, end_year, obs_countsi)
        obs_dates.append(
            np.array(['%d-06-15' % i for i in obs_yearsi], dtype='datetime64'))

    # add some earthquakes in incomplete years
    mags_inc = np.concatenate([
        np.random.randint(40, 50, 100) / 10,
        np.random.randint(50, 60, 10) / 10,
        np.random.randint(60, 70, 1) / 10,
    ])
    years_inc = np.concatenate([
        np.random.randint(1000, 1940, 100),
        np.random.randint(1000, 1880, 10),
        np.random.randint(1000, 1500, 1),
    ])
    dates_inc = np.array(
        ["%d-06-15" % i for i in years_inc], dtype="datetime64"
    )

    # merge complete and incomplete earthquakes
    mags = np.concatenate([*obs_mags, mags_inc])
    dates = np.concatenate([*obs_dates, dates_inc])
    return mags, dates


@pytest.mark.parametrize(
    "a_val_true,b_val_true,precision",
    [(7, 1, 0.01)]
)
def test_estimate_b_weichert(a_val_true: float,
                             b_val_true: float,
                             precision: float):
    mags, dates = _create_test_catalog_poisson(
        a_val_true, b_val_true)

    (b_val, std_b_val, rate_at_mref, std_rate_at_mref, a_val,) = \
        estimate_b_weichert(magnitudes=mags,
                            dates=dates,
                            completeness_table=np.array(
                                [[3.95, 1940],
                                 [4.95, 1880],
                                 [5.95, 1500],
                                 [6.95, 1000]]
                            ),
                            mag_max=7.95,
                            last_year=2000,
                            delta_m=0.1,
                            b_parameter="b_value",
                            )

    assert abs(b_val_true - b_val) / b_val_true <= precision
    assert abs(a_val_true - a_val) / a_val_true <= precision


# load data for test_shi_bolt_confidence
with open("seismostats/analysis/tests/data/test_shi_bolt_confidence.p",
          "rb") as f:
    data = pickle.load(f)


@pytest.mark.parametrize("magnitudes,b,b_parameter,std",
                         [data["values_test1"], data["values_test2"]],)
def test_shi_bolt_confidence(magnitudes: np.ndarray,
                             b: float, b_parameter: str,
                             std: float):
    precision = 1e-10
    assert (shi_bolt_confidence(magnitudes, b=b, b_parameter=b_parameter) - std
            < precision)
