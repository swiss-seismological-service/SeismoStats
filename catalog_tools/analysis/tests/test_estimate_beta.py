import pytest
import numpy as np

# import functions from other modules
from catalog_tools.utils.simulate_distributions import simulate_magnitudes
from catalog_tools.utils.binning import bin_to_precision

# import functions to be tested
from catalog_tools.analysis.estimate_beta import\
    estimate_beta_tinti,\
    estimate_b_tinti,\
    estimate_b_utsu,\
    estimate_b_elst,\
    estimate_b_laplace,\
    estimate_b_weichert,\
    differences,\
    shi_bolt_confidence


def simulate_magnitudes_w_offset(
        n: int, beta: float, mc: float,
        delta_m: float, mag_max: float = None) -> np.ndarray:
    """ This function simulates the magnitudes with the correct offset"""
    mags = simulate_magnitudes(n, beta, mc - delta_m / 2, mag_max)
    if delta_m > 0:
        mags = bin_to_precision(mags, delta_m)
    return mags


@pytest.mark.parametrize(
    "n,beta,mc,delta_m,precision",
    [(1000000, np.log(10), 3, 0, 0.005),
     (1000000, np.log(10), 3, 0.1, 0.01)]
)
def test_estimate_beta_tinti(n: int, beta: float, mc: float, delta_m: float,
                             precision: float):
    mags = simulate_magnitudes_w_offset(n, beta, mc, delta_m)
    beta_estimate = estimate_beta_tinti(mags, mc, delta_m)
    print((beta_estimate - beta) / beta)
    assert abs(beta - beta_estimate) / beta <= precision


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [(1000000, 1.2 * np.log(10), 3, 0, 'beta', 0.005),
     (1000000, np.log(10), 3, 0.1, 'beta', 0.01)]
)
def test_estimate_b_tinti(n: int, b: float, mc: float, delta_m: float,
                          b_parameter: str, precision: float):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_tinti(mags, mc, delta_m, b_parameter=b_parameter)
    print((b_estimate - b) / b)
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [(1000000, 1.2 * np.log(10), 3, 0, 'beta', 0.005),
     (1000000, np.log(10), 3, 0.1, 'beta', 0.01)]
)
def test_estimate_b_utsu(n: int, b: float, mc: float, delta_m: float,
                         b_parameter: str, precision: float):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_utsu(mags, mc, delta_m, b_parameter=b_parameter)
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "magnitudes,mag_diffs",
    [(np.array([1, -2, 3]),
      np.array([-3, 2, 3, 5, -2, -5]))]
)
def test_differences(magnitudes: np.ndarray, mag_diffs: np.ndarray):
    y = differences(magnitudes)
    assert (y == mag_diffs).all()


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [(1000000, 1.2 * np.log(10), 3, 0, 'beta', 0.005),
     (1000000, np.log(10), 3, 0.1, 'beta', 0.01)]
)
def test_estimate_b_elst(n: int, b: float, mc: float, delta_m: float,
                         b_parameter: str, precision: float):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_elst(mags, delta_m=delta_m, b_parameter=b_parameter)
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "n,b,mc,delta_m,b_parameter,precision",
    [(1000, 1.2 * np.log(10), 3, 0, 'beta', 0.15),
     (1000, np.log(10), 3, 0.1, 'beta', 0.2)]
)
def test_estimate_b_laplace(n: int, b: float, mc: float, delta_m: float,
                            b_parameter: str, precision: float):
    mags = simulate_magnitudes_w_offset(n, b, mc, delta_m)
    b_estimate = estimate_b_laplace(mags, delta_m=delta_m,
                                    b_parameter=b_parameter)
    assert abs(b - b_estimate) / b <= precision


@pytest.mark.parametrize(
    "a_val_true,b_val_true,precision",
    [(7, 1, 0.01)]
)
def test_estimate_b_weichert(a_val_true: float,
                             b_val_true: float,
                             precision: float):

    # annual expected rates:
    r45 = 10 ** (a_val_true - b_val_true * 3.95)\
        - 10 ** (a_val_true - b_val_true * 4.95)
    r56 = 10 ** (a_val_true - b_val_true * 4.95)\
        - 10 ** (a_val_true - b_val_true * 5.95)
    r67 = 10 ** (a_val_true - b_val_true * 5.95)\
        - 10 ** (a_val_true - b_val_true * 6.95)
    r78 = 10 ** (a_val_true - b_val_true * 6.95)\
        - 10 ** (a_val_true - b_val_true * 7.95)

    # assume a catalogue from year 1000 to end of 1999
    # with completeness as follows:
    # 3.95 - 1940 / 4.95 - 1880 / 5.95 - 1500 / 6.95 - 1000

    # sample earthquakes over 1,000 year period
    n45 = np.random.poisson(r45 * (2000 - 1940))
    mags45 = simulate_magnitudes_w_offset(
        n=n45, beta=np.log(10), mc=4, delta_m=0.1, mag_max=4.95)
    years45 = np.random.randint(1940, 2000, n45)
    dates45 = np.array(['%d-06-15' % i for i in years45], dtype='datetime64')

    n56 = np.random.poisson(r56 * (2000 - 1880))
    mags56 = simulate_magnitudes_w_offset(
        n=n56, beta=np.log(10), mc=5, delta_m=0.1, mag_max=5.95)
    years56 = np.random.randint(1880, 2000, n56)
    dates56 = np.array(['%d-06-15' % i for i in years56], dtype='datetime64')

    n67 = np.random.poisson(r67 * (2000 - 1500))
    mags67 = simulate_magnitudes_w_offset(
        n=n67, beta=np.log(10), mc=6, delta_m=0.1, mag_max=6.95)
    years67 = np.random.randint(1500, 2000, n67)
    dates67 = np.array(['%d-06-15' % i for i in years67], dtype='datetime64')

    n78 = np.random.poisson(r78 * (2000 - 1000))
    mags78 = simulate_magnitudes_w_offset(
        n=n78, beta=np.log(10), mc=7, delta_m=0.1, mag_max=7.95)
    years78 = np.random.randint(1000, 2000, n78)
    dates78 = np.array(['%d-06-15'%i for i in years78], dtype='datetime64')

    # add some earthquakes in incomplete years
    mags_inc = np.concatenate([
        np.random.randint(40, 50, 100) / 10,
        np.random.randint(50, 60, 10) / 10,
        np.random.randint(60, 70, 1) / 10
    ])
    years_inc = np.concatenate([
        np.random.randint(1000, 1940, 100),
        np.random.randint(1000, 1880, 10),
        np.random.randint(1000, 1500, 1)
    ])
    dates_inc = np.array(['%d-06-15' % i for i in years_inc], dtype='datetime64')

    mags = np.concatenate([mags45, mags56, mags67, mags78, mags_inc])
    dates = np.concatenate([dates45, dates56, dates67, dates78, dates_inc])

    b_val, std_b_val, rate_at_mref, std_rate_at_mref, a_val = \
        estimate_b_weichert(magnitudes=mags, dates=dates, completeness_table=np.array([[3.95, 1940], [4.95, 1880],
                                                                                       [5.95, 1500], [6.95, 1000]]),
                            mag_max=7.95, last_year=2000, delta_m=0.1, b_parameter='b_value')

    assert abs(b_val_true - b_val) / b_val_true <= precision
    assert abs(a_val_true - a_val) / a_val_true <= precision


@pytest.mark.parametrize(
    "magnitudes,b_value,std_b_value,std_beta",
    [(np.array([0.20990507, 0.04077336, 0.27906596, 0.57406287, 0.64256544,
                0.07293118, 0.58589873, 0.02678655, 0.27631233, 0.17682814]),
      1, 0.16999880611649493, 0.39143671679062625),
     (np.array([0.02637757, 0.06353823, 0.10257919, 0.54494906, 0.03928375,
                0.08825028, 0.77713586, 0.54553981, 0.69315583, 0.06656642,
                0.29035447, 0.2051877, 0.30858087, 0.68896342, 0.03328782,
                0.45016109, 0.40779409, 0.06788892, 0.02684032, 0.56140282,
                0.29443359, 0.36328762, 0.17124489, 0.02154936, 0.36461541,
                0.03613088, 0.15798366, 0.09111875, 0.16169287, 0.11986668,
                0.10232035, 0.72695761, 0.19484174, 0.0459675, 0.40514163,
                0.08979514, 0.0442659, 0.18672424, 0.21239088, 0.02287468,
                0.1244267, 0.04939361, 0.11232758, 0.02706083, 0.04275401,
                0.02326945, 0.15048133, 0.50777581, 0.09583551, 0.40618488,
                0.15595656, 0.09607254, 0.25576619, 0.01698973, 0.62755249,
                0.31429311, 0.86575907, 0.37956298, 0.65648246, 0.0851286,
                0.00850252, 0.22357953, 0.03295106, 0.08841752, 0.09657961,
                0.54002676, 0.20335658, 0.23215333, 0.20120566, 0.60970099,
                0.01128978, 0.31771308, 1.25246151, 0.02285632, 0.2687791,
                0.1192099, 0.06627574, 0.04301886, 0.24720467, 0.28518304,
                0.04252851, 0.27818821, 0.08331663, 1.23090656, 0.1880176,
                0.11314717, 0.01462853, 0.09256047, 0.4857446, 0.08656431,
                0.07022632, 0.32654491, 0.26047389, 0.20872121, 0.40157424,
                0.02732529, 0.83884229, 0.4147758, 0.07416183, 0.05636252]),
      1.5, 0.13286469044352858, 0.3059322556005374)]
)
def test_shi_bolt_confidence(
        magnitudes: np.ndarray,
        b_value: float,
        std_b_value: float,
        std_beta: float):
    precision = 1e-10
    beta = b_value * np.log(10)

    assert shi_bolt_confidence(
        magnitudes, b_value=b_value) - std_b_value < precision
    assert shi_bolt_confidence(magnitudes, beta=beta) - std_beta < precision
