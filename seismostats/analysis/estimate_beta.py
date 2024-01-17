"""This module contains functions for the estimation of beta and the b-value.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_b(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
    method="tinti",
    return_n: bool = False,
) -> float | tuple[float, float]:
    """returns the maximum likelihood beta or b-value. Method depends on the
    input parameter 'method'.

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no
                magnitudes below mc present)
        mc:         completeness magnitude
        delta_m:    discretization of magnitudes. default is no discretization
        weights:    weights of each magnitude can be specified here
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta'
                from the exponential distribution [p(M) = exp(-beta*(M-mc))]
        return_std: if True the standard deviation of beta/b-value (see
                above) is returned
        method:     method to use for estimation of beta/b-value. Options
                are:
                    - 'tinti', 'utsu' (these are the classic estimators. 'tinti'
                    is the recommended one, as it is more accurate. It is also
                    the default method)
                    - 'positive' (this is b-positive, which applies the 'tinti'
                    method to the positive differences. To achieve the effect
                    of reduced STAI, the magnitudes must be ordered in time)
                    - 'laplace' (this is using the distribution of all
                    differences, caution, can take long time to compute)

        return_n:   if True the number of events used for the estimation is
                returned. This is only relevant for the 'positive' method

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
            input variable 'gutenberg'. Note that the difference
            is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
        n:      number of events used for the estimation
    """

    # test that the magnitudes are binned correctly
    diffs = np.diff(magnitudes)
    assert (
        np.min(abs(diffs[diffs != 0])) - delta_m < delta_m * 1e-4
    ), "magnitudes are not binned correctly"
    # test that smallest magnitude is not below mc
    assert (
        np.min(magnitudes) >= mc
    ), "magnitudes below mc are present in the data"
    # test if lowest magnitude is much larger than mc
    if np.min(magnitudes) - mc > delta_m / 2:
        warnings.warn(
            "lowest magnitude is more than delta_m/2 larger than mc. "
            "Check if mc is chosen correctly"
        )

    if method == "tinti":
        return estimate_b_tinti(
            magnitudes,
            mc=mc,
            delta_m=delta_m,
            weights=weights,
            b_parameter=b_parameter,
            return_std=return_std,
        )
    elif method == "utsu":
        return estimate_b_utsu(
            magnitudes,
            mc=mc,
            delta_m=delta_m,
            b_parameter=b_parameter,
            return_std=return_std,
        )
    elif method == "positive":
        return estimate_b_positive(
            magnitudes,
            delta_m=delta_m,
            b_parameter=b_parameter,
            return_std=return_std,
            return_n=return_n,
        )
    elif method == "laplace":
        return estimate_b_laplace(
            magnitudes,
            delta_m=delta_m,
            b_parameter=b_parameter,
            return_std=return_std,
        )

    else:
        raise ValueError(
            "method must be either 'tinti', 'utsu', 'positive' or 'laplace'"
        )


def estimate_b_tinti(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
) -> float | tuple[float, float]:
    """returns the maximum likelihood beta
    Source:
        Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
            America, 77(6), 2125-2134.)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no
                    magnitudes below mc present)
        mc:         completeness magnitude
        delta_m:    discretization of magnitudes. default is no discretization
        weights:    weights of each magnitude can be specified here
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))]
        return_std: if True the standard deviation of beta/b-value (see above)
                is returned

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'gutenberg'. Note that the difference
                is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """

    if delta_m > 0:
        p = 1 + delta_m / np.average(magnitudes - mc, weights=weights)
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(magnitudes - mc, weights=weights)

    if b_parameter == "b_value":
        factor = 1 / np.log(10)
    elif b_parameter == "beta":
        factor = 1
    else:
        raise ValueError(
            "please choose either 'b_value' or 'beta' as b_parameter"
        )

    if return_std is True:
        std_b = shi_bolt_confidence(magnitudes, beta=beta) * factor
        b = beta * factor
        return b, std_b
    else:
        b = beta * factor
        return b


def estimate_b_utsu(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    b_parameter: str = "b_value",
    return_std: bool = False,
) -> float | tuple[float, float]:
    """returns the maximum likelihood beta
    Source:
        Utsu 1965 (Geophysical bulletin of the Hokkaido University, vol 13, pp
        99-103)

    Args:
        magnitudes: vector of magnitudes, unsorted, already cutoff (no
                    magnitudes below mc present)
        mc:         completeness magnitude
        delta_m:    discretization of magnitudes. default is no discretization
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))]
        return_std:  if True the standard deviation of beta/b-value (see above)
                is returned

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'gutenberg'. Note that the difference
                is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """
    beta = 1 / np.mean(magnitudes - mc + delta_m / 2)

    assert (
        b_parameter == "b_value" or b_parameter == "beta"
    ), "please choose either 'b_value' or 'beta' as b_parameter"
    if b_parameter == "b_value":
        factor = 1 / np.log(10)
    else:
        factor = 1

    if return_std is True:
        std_b = shi_bolt_confidence(magnitudes, beta=beta) * factor
        b = beta * factor
        return b, std_b
    else:
        b = beta * factor
        return b


def differences(magnitudes: np.ndarray) -> np.ndarray:
    """returns all the differences between the magnitudes.

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)

    Returns: array of all differences of the elements of the input
    """
    mag_diffs = np.array([])
    for ii, mag in enumerate(magnitudes):
        loop_mag = np.delete(magnitudes, [ii], axis=0)
        mag_diffs = np.append(mag_diffs, loop_mag - mag)
    return mag_diffs


def estimate_b_positive(
    magnitudes: np.ndarray,
    delta_m: float = 0,
    b_parameter: str = "b_value",
    return_std: bool = False,
    return_n: bool = False,
) -> float | tuple[float, float]:
    """returns the b-value estimation using the positive differences of the
    Magnitudes

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))]
        return_std: if True the standard deviation of beta/b-value (see above)
                is returned
        return_n:   if True the number of events used for the estimation is
                returned

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'gutenberg'. Note that the difference is just a
                factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """

    mag_diffs = np.diff(magnitudes)
    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs > 0])

    out = estimate_b_tinti(
        mag_diffs,
        mc=delta_m,
        delta_m=delta_m,
        b_parameter=b_parameter,
        return_std=return_std,
    )

    if return_n:
        return out + tuple([len(mag_diffs)])
    else:
        return out


def estimate_b_laplace(
    magnitudes: np.ndarray,
    delta_m: float = 0,
    b_parameter: str = "b_value",
    return_std: bool = False,
) -> float | tuple[float, float]:
    """returns the b-value estimation using the all the  differences of the
    Magnitudes (this has a little less variance than the estimate_b_positive
    method)

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))]
        return_std: if True the standard deviation of beta/b-value (see above)
                is returned

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'b_parameter'. Note that the difference is just a
                factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """
    mag_diffs = differences(magnitudes)
    mag_diffs = abs(mag_diffs)
    mag_diffs = mag_diffs[mag_diffs > 0]
    return estimate_b_tinti(
        mag_diffs,
        mc=delta_m,
        delta_m=delta_m,
        b_parameter=b_parameter,
        return_std=return_std,
    )


def estimate_b_weichert(
    magnitudes: np.ndarray,
    dates: list[np.datetime64],
    completeness_table: np.ndarray,
    mag_max: int | float,
    last_year: int | float | None = None,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> tuple[float, float, float, float, float]:
    """applies the Weichert (1980) algorithm for estimation of the
    Gutenberg-Richter magnitude-frequency distribution parameters in
    the case of unequal completeness periods for different magnitude
    values.

    Source:
        Weichert (1980), Estimation of the earthquake recurrence parameters
        for unequal observation periods for different magnitudes,
        Bulletin of the Seismological Society of America,
        Vol 70, No. 4, pp. 1337-1346

    Args:
        magnitudes: vector of earthquake magnitudes
        dates: list of datetime objects of occurrence of each earthquake
        completeness_table: Nx2 array, where the first column
            contains the leftmost edge of magnitude bins and
            the second column the associated year of completeness, i.e.
            the year after which all earthquakes larger than the value in
            the first column are considered detected. An example is given
            below:

            np.array([[ 3.95, 1980],
                      [ 4.95, 1920],
                      [ 5.95, 1810],
                      [ 6.95, 1520]])

        mag_max: maximum possible magnitude
        last_year: last year of observation (the default is None, in which case
              it is set to the latest year in years).
        delta_m: magnitude resolution, the default is 0.1.
        b_parameter:either 'b-value', then the corresponding value of the
                    Gutenberg-Richter law is returned, otherwise 'beta'
                    from the exponential distribution [p(M) = exp(-beta*(M-mc))]

    Returns:(
        b_parameter: maximum likelihood point estimate of 'b-value' or 'beta'
        std_b_parameter: standard error of b_parameter
        rate_at_lmc: maximum likelihood point estimate of earthquake rate
                     at the lower magnitude of completeness
        std_rate_at_lmc: standard error of rate_at_lmc
        a_val: maximum likelihood point estimate of a-value
               ( =log10(rate at mag=0) ) of Gutenberg-Richter
               magnitude frequency distribution
    """
    assert len(magnitudes) == len(
        dates
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
    if not np.all(magnitudes >= completeness_table[:, 0].min()):
        warnings.warn(
            "magnitudes below %.2f are not covered by the "
            "completeness table and are discarded" % completeness_table[0, 0]
        )
    assert delta_m > 0, "delta_m cannot be zero"
    assert (
        b_parameter == "b_value" or b_parameter == "beta"
    ), "please choose either 'b_value' or 'beta' as b_parameter"
    factor = 1 / np.log(10) if b_parameter == "b_value" else 1

    # convert datetime to integer calendar year
    years = np.array(dates).astype("datetime64[Y]").astype(int) + 1970

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
            completeness_table_years[idx - 1]
            if idx not in [0, len(completeness_table_years)]
            else {
                0: -1,
                len(completeness_table_years): completeness_table_years[-1],
            }[idx]
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
    b_parameter = beta * factor

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
    std_b_parameter = np.sqrt(var_beta) * factor

    # compute uncertainty in rate at lower magnitude of completeness
    std_rate_at_lmc = rate_at_lmc / np.sqrt(complete_events.num.sum())

    return b_parameter, std_b_parameter, rate_at_lmc, std_rate_at_lmc, a_val


def _weichert_objective_function(
    beta: float,
    last_year: float | int,
    complete_events: pd.DataFrame,
    delta_m: float,
) -> float:
    """
    function to be minimized for estimation of GR parameters as per
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


def shi_bolt_confidence(
    magnitudes: np.ndarray,
    b_value: float | None = None,
    beta: float | None = None,
) -> float:
    """calculates the confidence limit of the b_value or beta (depending on
        which parameter is given) according to shi and bolt 1982

    Source:
        Shi and Bolt, BSSA, Vol. 72, No. 5, pp. 1677-1687, October 1982

    Args:
        magnitudes: numpy array of magnitudes
        b_value:    b-value of the magnitudes
        beta:       beta value (difference to b-value is factor of np.log(10)).
                    -> provide either b_value or beta, not both

    Returns:
        sig_b:  confidence limit of the b-value/beta value (depending on input)
    """
    # standard deviation in Shi and Bolt is calculated with 1/(N*(N-1)), which
    # is by a factor of sqrt(N) different to the std(x, ddof=1) estimator
    assert (
        b_value is not None or beta is not None
    ), "please specify b-value or beta"
    assert (
        b_value is None or beta is None
    ), "please only specify either b-value or beta"

    if b_value is not None:
        std_m = np.std(magnitudes, ddof=1) / np.sqrt(len(magnitudes))
        std_b = np.log(10) * b_value**2 * std_m
    else:
        std_m = np.std(magnitudes, ddof=1) / np.sqrt(len(magnitudes))
        std_b = beta**2 * std_m

    return std_b
