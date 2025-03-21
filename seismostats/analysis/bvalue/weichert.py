
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from seismostats.analysis.bvalue.utils import beta_to_b_value
from seismostats.utils._config import get_option


def estimate_b_weichert(
    magnitudes: np.ndarray,
    times: list[np.datetime64],
    completeness_table: np.ndarray,
    mag_max: int | float,
    last_year: int | float | None = None,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> tuple[float, float, float, float, float]:
    """
    Returns the b-value of the Gutenberg-Richter law calculated using the
    Weichert (1980) algorithm, for the case of unequal
    completeness periods for different magnitude values.

    Source:
        Weichert (1980), Estimation of the earthquake recurrence parameters
        for unequal observation periods for different magnitudes,
        Bulletin of the Seismological Society of America,
        Vol 70, No. 4, pp. 1337-1346

    Args:
        magnitudes:         Array of magnitudes.
        dates:              Array of datetime objects of earthquake occurrences.
        completeness_table: Nx2 array, where the first column
            contains the leftmost edge of magnitude bins and
            the second column the associated year of completeness, i.e.
            the year after which all earthquakes larger than the value in
            the first column are considered detected. An example is given
            below.

            >>> np.array([[ 3.95, 1980],
            ...          [ 4.95, 1920],
            ...          [ 5.95, 1810],
            ...          [ 6.95, 1520]])

        mag_max:        Maximum possible magnitude.
        last_year:      Last year of observation (the default is None, in which
              case it is set to the latest year in years).
        delta_m:        Bin size of discretized magnitudes.
        b_parameter:    Either 'b-value', then the corresponding value of the
                    Gutenberg-Richter law is returned, otherwise 'beta' from
                    the exponential distribution [p(M) = exp(-beta*(M-mc))].

    Returns:
        b_parameter:        Maximum likelihood estimate of 'b-value' or 'beta'.
        std_b_parameter:    Standard error of b_parameter.
        rate_at_lmc:        Maximum likelihood point estimate of earthquake rate
                     at the lower magnitude of completeness.
        std_rate_at_lmc:    Standard error of rate_at_lmc.
        a_val:              Maximum likelihood point estimate of a-value
               ( =log10(rate at mag=0) ) of Gutenberg-Richter
               magnitude frequency distribution.
    """
    assert len(magnitudes) == len(
        times
    ), "the magnitudes and years arrays have different lengths"
    assert completeness_table.shape[1] == 2
    assert np.all(
        np.ediff1d(completeness_table[:, 0]) >= 0
    ), "magnitudes in completeness table not in ascending order"
    assert [
        i - delta_m
        in np.arange(completeness_table[0, 0], mag_max + 0.001, delta_m)
        for i in np.unique(magnitudes)
    ], "magnitude bins not aligned with completeness edges"
    if get_option("warnings") is True:
        if not np.all(magnitudes >= completeness_table[:, 0].min()):
            warnings.warn(
                "magnitudes below %.2f are not covered by the "
                "completeness table and are discarded"
                % completeness_table[0, 0]
            )
    assert delta_m > 0, "delta_m cannot be zero"
    assert (
        b_parameter == "b_value" or b_parameter == "beta"
    ), "please choose either 'b_value' or 'beta' as b_parameter"

    # convert datetime to integer calendar year
    years = np.array(times).astype("datetime64[Y]").astype(int) + 1970

    # get last year of catalogue if last_year not defined
    last_year = last_year if last_year else np.max(years)

    # Get the magnitudes and completeness years as separate arrays
    completeness_table_magnitudes = completeness_table[:, 0]
    completeness_table_years = completeness_table[:, 1]

    # Obtain the completeness start year for each value in magnitudes
    insertion_indices = np.searchsorted(
        completeness_table_magnitudes, magnitudes
    )
    completeness_starts = np.array(
        [
            (
                completeness_table_years[idx - 1]
                if idx not in [0, len(completeness_table_years)]
                else {
                    0: -1,
                    len(completeness_table_years): completeness_table_years[
                        -1
                    ],
                }[idx]
            )
            for i, idx in enumerate(insertion_indices)
        ]
    )

    # filter out events outside completeness window and
    # get number of "complete" events in each magnitude bin
    # and associated year of completeness
    idxcomp = (completeness_starts > 0) & (years - completeness_starts >= 0)
    complete_events = (
        pd.DataFrame.groupby(
            pd.DataFrame(
                data={
                    "mag_left_edge": np.array(
                        [
                            i.left
                            for i in pd.cut(
                                magnitudes[idxcomp],
                                bins=np.arange(
                                    completeness_table_magnitudes[0],
                                    mag_max + 0.01,
                                    delta_m,
                                ),
                                right=False,
                            )
                        ]
                    ),
                    "completeness_start": completeness_starts[idxcomp],
                }
            ),
            by=["mag_left_edge", "completeness_start"],
        )
        .size()
        .to_frame("num")
        .reset_index()
    )
    assert np.all(
        complete_events.completeness_start > 0
    )  # should be the case by design

    # minimization
    beta = np.log(10)  # initialization of beta
    solution = minimize(
        _weichert_objective_function,
        beta,
        args=(last_year, complete_events, delta_m),
        method="Nelder-Mead",
        options={"maxiter": 5000, "disp": True},
        tol=1e5 * np.finfo(float).eps,
    )
    beta = solution.x[0]

    # compute rate at lower magnitude of completeness bin
    weichert_multiplier = np.sum(
        np.exp(-beta * (complete_events.mag_left_edge + delta_m * 0.5))
    ) / np.sum(
        (last_year - complete_events.completeness_start.values)
        * np.exp(-beta * (complete_events.mag_left_edge + delta_m * 0.5))
    )
    rate_at_lmc = complete_events.num.sum() * weichert_multiplier

    # compute a-value ( a_val = log10(rate at M=0) )
    a_val = (
        np.log10(rate_at_lmc)
        + (beta / np.log(10)) * complete_events.mag_left_edge.values[0]
    )

    # compute uncertainty in b-parameter according to Weichert (1980)
    nominator = (
        np.sum(
            (last_year - complete_events.completeness_start.values)
            * np.exp(-beta * (complete_events.mag_left_edge + delta_m * 0.5))
        )
        ** 2
    )
    denominator_term1 = (
        np.sum(
            (last_year - complete_events.completeness_start.values)
            * (complete_events.mag_left_edge + delta_m * 0.5)
            * np.exp(-beta * (complete_events.mag_left_edge + delta_m * 0.5))
        )
        ** 2
    )
    denominator_term2 = np.sqrt(nominator) * np.sum(
        (last_year - complete_events.completeness_start.values)
        * ((complete_events.mag_left_edge + delta_m * 0.5) ** 2)
        * np.exp(-beta * (complete_events.mag_left_edge + delta_m * 0.5))
    )
    var_beta = (
        -(1 / complete_events.num.sum())
        * nominator
        / (denominator_term1 - denominator_term2)
    )
    std_b = np.sqrt(var_beta)
    if b_parameter == "b_value":
        b = beta_to_b_value(beta)
        std_b = beta_to_b_value(std_b)

    # compute uncertainty in rate at lower magnitude of completeness
    std_rate_at_lmc = rate_at_lmc / np.sqrt(complete_events.num.sum())

    return b, std_b, rate_at_lmc, std_rate_at_lmc, a_val


def _weichert_objective_function(
    beta: float,
    last_year: float | int,
    complete_events: pd.DataFrame,
    delta_m: float,
) -> float:
    """
    Function to be minimized for estimation of GR parameters as per
    Weichert (1980). Used internally within estimate_b_weichert function.
    """
    magbins = complete_events.mag_left_edge + delta_m * 0.5
    nom = np.sum(
        (last_year - complete_events.completeness_start.values)
        * magbins
        * np.exp(-beta * magbins)
    )
    denom = np.sum(
        (last_year - complete_events.completeness_start.values)
        * np.exp(-beta * magbins)
    )
    left = nom / denom
    right = (
        np.sum(complete_events.num.values * magbins)
        / complete_events.num.sum()
    )
    return np.abs(left - right)
