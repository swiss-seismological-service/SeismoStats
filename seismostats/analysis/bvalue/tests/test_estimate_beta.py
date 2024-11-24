import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from seismostats.analysis.bvalue import (BMorePositiveEstimator,
                                         BPositiveEstimator,
                                         ClassicBValueEstimator,
                                         UtsuBValueEstimator, estimate_b)
# import functions to be tested
from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               make_more_incomplete,
                                               shi_bolt_confidence)
from seismostats.analysis.bvalue.weichert import estimate_b_weichert
from seismostats.utils.simulate_distributions import (
    bin_to_precision, simulate_magnitudes_binned)


@pytest.mark.parametrize(
    "method, return_std, return_n, b_parameter",
    [
        ("classic", True, True, "beta"),
        ("classic", False, False, "b_value"),
        ("positive", True, True, "beta"),
        ("positive", False, False, "b_value"),
        ("positive", True, False, "beta"),
        ("positive", False, True, "b_value"),
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
    mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
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
    assert np.size(out) == 1 + return_std + (return_n * (method == "positive"))

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

    # test that magnitudes smaller than mc lead to error
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


# def magnitudes(b: float):
#     df_mags = pd.read_csv(
#         'seismostats/analysis/tests/data/simulated_magnitudes.csv')
#     if b == 0.5:
#         mags = df_mags["b_value = 0.5"].values
#     elif b == 1:
#         mags = df_mags["b_value = 1"].values
#     elif b == 1.5:
#         mags = df_mags["b_value = 1.5"].values
#     return mags


# @pytest.mark.parametrize(
#     "b_est_correct, mags, mc, delta_m, b_parameter",
#     [
#         (0.9985052730956719, magnitudes(1), 0, 0.1, "b_value"),
#         (1.486114976626299, magnitudes(1.5), 0.5, 0.01, "b_value"),
#         (1.15416174113439, magnitudes(0.5), 2, 0.2, "beta"),
#     ],
# )
# def test_estimate_b_classic(
#     b_est_correct: float,
#     mags: np.ndarray,
#     mc: float,
#     delta_m: float,
#     b_parameter: str,
# ):
#     mags = bin_to_precision(mags, delta_m)
#     mags = mags[mags >= mc - delta_m / 2]
#     b_estimate = estimate_b_classic(mags, mc, delta_m, b_parameter=b_parameter)

#     assert_almost_equal(b_estimate, b_est_correct)


# @pytest.mark.parametrize(
#     "b_est_correct, mags, mc, delta_m, b_parameter",
#     [
#         (0.9941299341459253, magnitudes(1), 0, 0.1, "b_value"),
#         (1.485969980462011, magnitudes(1.5), 0.5, 0.01, "b_value"),
#         (1.149064079792321, magnitudes(0.5), 2, 0.2, "beta"),
#     ],
# )
# def test_estimate_b_utsu(
#     b_est_correct: float,
#     mags: np.ndarray,
#     mc: float,
#     delta_m: float,
#     b_parameter: str,
# ):
#     mags = bin_to_precision(mags, delta_m)
#     mags = mags[mags >= mc - delta_m / 2]
#     b_estimate = estimate_b_utsu(mags, mc, delta_m, b_parameter=b_parameter)
#     assert_almost_equal(b_estimate, b_est_correct)


# @pytest.mark.parametrize(
#     "b_est_correct, mags, mc, delta_m, dmc, b_parameter",
#     [
#         (1.00768483769521, magnitudes(1), 0, 0.1, 0.3, "b_value"),
#         (1.4946439854664, magnitudes(1.5), 0.5, 0.01, None, "b_value"),
#         (1.129176714899606, magnitudes(0.5), 2, 0.2, None, "beta"),
#     ],
# )
# def test_estimate_b_positive(
#     b_est_correct: float,
#     mags: np.ndarray,
#     mc: float,
#     delta_m: float,
#     dmc: float,
#     b_parameter: str,
# ):
#     mags = bin_to_precision(mags, delta_m)
#     mags = mags[mags >= mc - delta_m / 2]
#     b_estimate = estimate_b_positive(
#         mags, delta_m=delta_m, dmc=dmc, b_parameter=b_parameter
#     )
#     assert_almost_equal(b_estimate, b_est_correct)


# @pytest.mark.parametrize(
#     "b_est_correct, mags, mc, delta_m, dmc, b_parameter",
#     [
#         (1.03259579513585, magnitudes(1), 0, 0.1, 0.3, "b_value"),
#         (1.476841984167775, magnitudes(1.5), 0.5, 0.01, None, "b_value"),
#         (1.121139770476674, magnitudes(0.5), 2, 0.2, None, "beta"),
#     ],
# )
# def test_estimate_b_more_positive(
#     b_est_correct: float,
#     mags: np.ndarray,
#     mc: float,
#     delta_m: float,
#     dmc: float,
#     b_parameter: str,
# ):
#     mags = bin_to_precision(mags, delta_m)
#     mags = mags[mags >= mc - delta_m / 2]
#     b_estimate = estimate_b_more_positive(
#         mags, delta_m=delta_m, dmc=dmc, b_parameter=b_parameter
#     )
#     assert_almost_equal(b_estimate, b_est_correct)


# def test_make_more_incomplete():
#     magnitudes = np.array([1, 2, 20, 3, 4, 9, 3])
#     times = np.array([
#         dt.datetime(2020, 1, 1),
#         dt.datetime(2020, 1, 2),
#         dt.datetime(2020, 1, 3),
#         dt.datetime(2020, 1, 4),
#         dt.datetime(2020, 1, 5),
#         dt.datetime(2020, 1, 6),
#         dt.datetime(2020, 1, 7),
#     ])

#     mags_inc, times_inc = make_more_incomplete(
#         magnitudes, times, delta_t=np.timedelta64(49, "h")
#     )

#     assert (mags_inc == [1, 2, 20, 9]).all()
#     assert (
#         times_inc
#         == [
#             dt.datetime(2020, 1, 1),
#             dt.datetime(2020, 1, 2),
#             dt.datetime(2020, 1, 3),
#             dt.datetime(2020, 1, 6),
#         ]
#     ).all()

#     mags_inc, times_inc, idx = make_more_incomplete(
#         magnitudes, times, delta_t=np.timedelta64(49, "h"), return_idx=True
#     )

#     assert (mags_inc == magnitudes[idx]).all()


# def _create_test_catalog_poisson(a_val_true: float, b_val_true: float):
#     """
#     creates a synthetic catalog with magnitudes
#     between 4 and 7.9 with unequal completeness periods. To be used
#     for testing relevant recurrence parameter estimators.
#     """

#     # assume a catalog from year 1000 to end of 1999
#     # with completeness as follows:
#     completeness_table = np.array(
#         [[3.95, 1940], [4.95, 1880], [5.95, 1500], [6.95, 1000]]
#     )

#     end_year = 2000
#     mmax = 7.95

#     obs_mags = []
#     obs_times = []
#     for ii in range(len(completeness_table)):
#         bin_lower_edge, cyear_lower = completeness_table[ii]
#         if ii == len(completeness_table) - 1:
#             bin_upper_edge = mmax
#         else:
#             bin_upper_edge, _ = completeness_table[ii + 1]

#         # get expected annual rates in completeness bin
#         exp_rate = 10 ** (a_val_true - b_val_true * bin_lower_edge) - 10 ** (
#             a_val_true - b_val_true * bin_upper_edge
#         )

#         # sample observed earthquakes over 1,000 year period
#         obs_countsi = np.random.poisson(exp_rate * (end_year - cyear_lower))
#         obs_mags.append(
#             simulate_magnitudes_binned(
#                 n=obs_countsi,
#                 b=1,
#                 mc=bin_lower_edge + 0.05,
#                 delta_m=0.1,
#                 mag_max=bin_upper_edge,
#             )
#         )
#         obs_yearsi = np.random.randint(cyear_lower, end_year, obs_countsi)
#         obs_times.append(
#             np.array(["%d-06-15" % i for i in obs_yearsi], dtype="datetime64")
#         )

#     # add some earthquakes in incomplete years
#     mags_inc = np.concatenate(
#         [
#             np.random.randint(40, 50, 100) / 10,
#             np.random.randint(50, 60, 10) / 10,
#             np.random.randint(60, 70, 1) / 10,
#         ]
#     )
#     years_inc = np.concatenate(
#         [
#             np.random.randint(1000, 1940, 100),
#             np.random.randint(1000, 1880, 10),
#             np.random.randint(1000, 1500, 1),
#         ]
#     )
#     times_inc = np.array(
#         ["%d-06-15" % i for i in years_inc], dtype="datetime64"
#     )

#     # merge complete and incomplete earthquakes
#     mags = np.concatenate([*obs_mags, mags_inc])
#     times = np.concatenate([*obs_times, times_inc])
#     return mags, times


# @pytest.mark.parametrize("a_val_true,b_val_true,precision", [(7, 1, 0.01)])
# def test_estimate_b_weichert(
#     a_val_true: float, b_val_true: float, precision: float
# ):
#     mags, times = _create_test_catalog_poisson(a_val_true, b_val_true)

#     (
#         b_val,
#         std_b_val,
#         rate_at_mref,
#         std_rate_at_mref,
#         a_val,
#     ) = estimate_b_weichert(
#         magnitudes=mags,
#         times=times,
#         completeness_table=np.array(
#             [[3.95, 1940], [4.95, 1880], [5.95, 1500], [6.95, 1000]]
#         ),
#         mag_max=7.95,
#         last_year=2000,
#         delta_m=0.1,
#         b_parameter="b_value",
#     )

#     assert abs(b_val_true - b_val) / b_val_true <= precision
#     assert abs(a_val_true - a_val) / a_val_true <= precision


# @pytest.mark.parametrize(
#     "std, mags, b, b_parameter",
#     [
#         (0.09776728958456313, magnitudes(1)[:100], 1, "b_value"),
#         (0.1062329763800726, magnitudes(1.5)[:200], 1.5, "b_value"),
#         (0.100184931569467, magnitudes(0.5)[
#          :100], b_value_to_beta(0.5), "beta"),
#     ],
# )
# def test_shi_bolt_confidence(
#         std: float, mags: np.ndarray, b: float, b_parameter: str):
#     assert_almost_equal(
#         shi_bolt_confidence(mags, b=b, b_parameter=b_parameter), std)
