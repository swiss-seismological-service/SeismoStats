import numpy as np
import pytest
import pickle

# import functions to be tested
from seismostats.analysis.estimate_beta import (
    differences,
    estimate_b,
    estimate_b_positive,
    estimate_b_laplace,
    estimate_b_tinti,
    estimate_b_utsu,
    estimate_b_weichert,
    shi_bolt_confidence,
)
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


@pytest.mark.parametrize("a_val_true,b_val_true,precision", [(7, 1, 0.01)])
def test_estimate_b_weichert(
    a_val_true: float, b_val_true: float, precision: float
):
    # annual expected rates:
    r45 = 10 ** (a_val_true - b_val_true * 3.95) - 10 ** (
        a_val_true - b_val_true * 4.95
    )
    r56 = 10 ** (a_val_true - b_val_true * 4.95) - 10 ** (
        a_val_true - b_val_true * 5.95
    )
    r67 = 10 ** (a_val_true - b_val_true * 5.95) - 10 ** (
        a_val_true - b_val_true * 6.95
    )
    r78 = 10 ** (a_val_true - b_val_true * 6.95) - 10 ** (
        a_val_true - b_val_true * 7.95
    )

    # assume a catalogue from year 1000 to end of 1999
    # with completeness as follows:
    # 3.95 - 1940 / 4.95 - 1880 / 5.95 - 1500 / 6.95 - 1000

    # sample earthquakes over 1,000 year period
    n45 = np.random.poisson(r45 * (2000 - 1940))
    mags45 = simulate_magnitudes_w_offset(
        n=n45, beta=np.log(10), mc=4, delta_m=0.1, mag_max=4.95
    )
    years45 = np.random.randint(1940, 2000, n45)
    dates45 = np.array(["%d-06-15" % i for i in years45], dtype="datetime64")

    n56 = np.random.poisson(r56 * (2000 - 1880))
    mags56 = simulate_magnitudes_w_offset(
        n=n56, beta=np.log(10), mc=5, delta_m=0.1, mag_max=5.95
    )
    years56 = np.random.randint(1880, 2000, n56)
    dates56 = np.array(["%d-06-15" % i for i in years56], dtype="datetime64")

    n67 = np.random.poisson(r67 * (2000 - 1500))
    mags67 = simulate_magnitudes_w_offset(
        n=n67, beta=np.log(10), mc=6, delta_m=0.1, mag_max=6.95
    )
    years67 = np.random.randint(1500, 2000, n67)
    dates67 = np.array(["%d-06-15" % i for i in years67], dtype="datetime64")

    n78 = np.random.poisson(r78 * (2000 - 1000))
    mags78 = simulate_magnitudes_w_offset(
        n=n78, beta=np.log(10), mc=7, delta_m=0.1, mag_max=7.95
    )
    years78 = np.random.randint(1000, 2000, n78)
    dates78 = np.array(["%d-06-15" % i for i in years78], dtype="datetime64")

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
    dates_inc = np.array(
        ["%d-06-15" % i for i in years_inc], dtype="datetime64"
    )

    mags = np.concatenate([mags45, mags56, mags67, mags78, mags_inc])
    dates = np.concatenate([dates45, dates56, dates67, dates78, dates_inc])

    (
        b_val,
        std_b_val,
        rate_at_mref,
        std_rate_at_mref,
        a_val,
    ) = estimate_b_weichert(
        magnitudes=mags,
        dates=dates,
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


# load data for test_shi_bolt_confidence
with open(
    "seismostats/analysis/tests/data/test_shi_bolt_confidence.p", "rb"
) as f:
    data = pickle.load(f)


@pytest.mark.parametrize(
    "magnitudes,b,b_parameter,std",
    [
        data["values_test1"],
        data["values_test2"],
    ],
)
def test_shi_bolt_confidence(
    magnitudes: np.ndarray, b: float, b_parameter: str, std: float
):
    precision = 1e-10
    assert (
        shi_bolt_confidence(magnitudes, b=b, b_parameter=b_parameter) - std
        < precision
    )
