"""This module contains functions for the estimation of beta and the b-value.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime as dt

from seismostats.utils._config import get_option


def beta_to_b_value(beta: float) -> float:
    """converts the beta value to the b-value  of the Gutenberg-Richter law

    Args:
        beta: beta value

    Returns:
        b_value: corresponding b-value
    """
    return beta / np.log(10)


def b_value_to_beta(b_value: float) -> float:
    """converts the b-value to the beta value of the exponential distribution

    Args:
        b_value: b-value

    Returns:
        beta: corresponding beta value
    """
    return b_value * np.log(10)


def estimate_b(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
    method="tinti",
    return_n: bool = False,
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Return the maximum likelihood estimator for the Gutenberg-Richter
    b-value or beta, given an array of magnitudes
    and a completeness magnitude.
    Estimation method depends on the input parameter ``method``.

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

                - 'tinti',default, this is the is the classic estimator, see
                  :func:`seismostats.analysis.estimate_b_tinti`
                - 'positive' (this is b-positive, which applies the 'tinti'
                  method to the positive differences, see
                  :func:`seismostats.analysis.estimate_b_positive`. To
                  achieve the effect of reduced STAI, the magnitudes must
                  be ordered in time)
        return_n:   if True the number of events used for the estimation is
                returned. This is only relevant for the 'positive' method

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
            input variable 'b_parameter'. Note that the difference
            is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate error
        n:      number of events used for the estimation
    """

    # test that the magnitudes are binned correctly
    mags_unique = np.unique(magnitudes)
    assert (
        max((mags_unique / delta_m) - np.round(mags_unique / delta_m)) < 1e-4
    ), "magnitudes are not binned correctly"
    # test that smallest magnitude is not below mc
    assert (
        np.min(magnitudes) >= mc
    ), "magnitudes below mc are present in the data"
    # test if lowest magnitude is much larger than mc
    if get_option("warnings") is True:
        if np.min(magnitudes) - mc > delta_m / 2:
            warnings.warn(
                "no magnitudes in the lowest magnitude bin are present."
                "check if mc is chosen correctly"
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

    elif method == "positive":
        return estimate_b_positive(
            magnitudes,
            delta_m=delta_m,
            b_parameter=b_parameter,
            return_std=return_std,
            return_n=return_n,
        )

    else:
        raise ValueError("method must be either 'tinti' or 'positive'")


def estimate_b_tinti(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
) -> float | tuple[float, float]:
    """Return the maximum likelihood b-value or beta for
    an array of magnitudes and a completeness magnitude mc.
    If the magnitudes are discretized, the discretization must be given in
    ``delta_m``, so that the maximum likelihood estimator can be calculated
    correctly.


    Source:
        - Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        - Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
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
                input variable 'b_parameter'. Note that the difference
                is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """

    if delta_m > 0:
        p = 1 + delta_m / np.average(magnitudes - mc, weights=weights)
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(magnitudes - mc, weights=weights)

    if b_parameter == "b_value":
        b = beta_to_b_value(beta)
    elif b_parameter == "beta":
        b = beta
    else:
        raise ValueError(
            "please choose either 'b_value' or 'beta' as b_parameter"
        )

    std = shi_bolt_confidence(magnitudes, b=b, b_parameter=b_parameter)

    if return_std is True:
        return b, std
    else:
        return b


def estimate_b_utsu(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    b_parameter: str = "b_value",
    return_std: bool = False,
) -> float | tuple[float, float]:
    """Return the maximum likelihood b-value or beta.

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
                input variable 'b_parameter'. Note that the difference
                is just a factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
    """
    beta = 1 / np.mean(magnitudes - mc + delta_m / 2)

    if b_parameter == "b_value":
        b = beta_to_b_value(beta)
    elif b_parameter == "beta":
        b = beta
    else:
        raise ValueError(
            "please choose either 'b_value' or 'beta' as b_parameter"
        )

    std = shi_bolt_confidence(magnitudes, b=b, b_parameter=b_parameter)

    if return_std is True:
        return b, std
    else:
        return b


def differences(magnitudes: np.ndarray) -> np.ndarray:
    """returns all the differences between the magnitudes, only counting each
    difference once

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)

    Returns: array of all differences of the elements of the input
    """
    mag_diffs = np.array([])
    for ii in range(1, len(magnitudes)):
        loop_mag1 = magnitudes[ii:]
        loop_mag2 = magnitudes[:-ii]
        mag_diffs = np.append(mag_diffs, loop_mag1 - loop_mag2)
    return mag_diffs


def estimate_b_positive(
    magnitudes: np.ndarray,
    delta_m: float = 0,
    dmc: float | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
    return_n: bool = False,
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Return the b-value estimate calculated using the
    positive differences between consecutive magnitudes.

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: array of magnitudes, sorted in time (first
                    entry is the earliest earthquake).
                    To achieve the effect
                    of reduced STAI, the magnitudes must be ordered in time.
        delta_m:    discretization of magnitudes. default is no discretization.
        dmc:       cutoff value for the differences (diffferences below this
                value are not considered). If None, the cutoff is set to delta_m
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))].
        return_std: if True the standard deviation of beta/b-value (see above)
                is returned.
        return_n:   if True the number of events used for the estimation is
                returned.

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'b_parameter'. Note that the difference is just a
                factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
        n:      number of events used for the estimation
    """

    if dmc is None:
        dmc = delta_m

    mag_diffs = np.diff(magnitudes)
    # only take the values where the next earthquake is d_mc larger than the
    # previous one. delta_m is added to avoid numerical errors
    mag_diffs = abs(mag_diffs[mag_diffs > dmc - delta_m / 2])

    out = estimate_b_tinti(
        mag_diffs,
        mc=dmc,
        delta_m=delta_m,
        b_parameter=b_parameter,
        return_std=return_std,
    )

    if return_n:
        if type(out) is tuple:
            return out + tuple([len(mag_diffs)])
        else:
            return out, len(mag_diffs)
    else:
        return out


def estimate_b_more_positive(
    magnitudes: np.ndarray,
    delta_m: float = 0,
    dmc: float | None = None,
    b_parameter: str = "b_value",
    return_std: bool = False,
    return_n: bool = False,
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Return the b-value estimate calculated using the
    next positive differences (this means that almost every magnitude has a
    difference, as opposed to the b-positive method which results in half the
    data).

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Args:
        magnitudes: array of magnitudes, sorted in time (first
                    entry is the earliest earthquake).
                    To achieve the effect
                    of reduced STAI, the magnitudes must be ordered in time.
        delta_m:    discretization of magnitudes. default is no discretization.
        b_parameter:either 'b-value', then the corresponding value  of the
                Gutenberg-Richter law is returned, otherwise 'beta' from the
                exponential distribution [p(M) = exp(-beta*(M-mc))].
        return_std: if True the standard deviation of beta/b-value (see above)
                is returned.
        return_n:   if True the number of events used for the estimation is
                returned.

    Returns:
        b:      maximum likelihood beta or b-value, depending on value of
                input variable 'b_parameter'. Note that the difference is just a
                factor [b_value = beta * log10(e)]
        std:    Shi and Bolt estimate of the beta/b-value estimate
        n:      number of events used for the estimation
    """

    if dmc is None:
        dmc = delta_m

    mag_diffs = np.zeros(len(magnitudes) - 1)
    for ii in range(len(magnitudes) - 1):
        for jj in range(ii + 1, len(magnitudes)):
            mag_diff_loop = magnitudes[jj] - magnitudes[ii]
            # print(mag_diff_loop, "diff loop")
            if mag_diff_loop > dmc - delta_m / 2:
                mag_diffs[ii] = mag_diff_loop
                # print("take the value")
                break

    # print(mag_diffs)

    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs > 0])

    out = estimate_b_tinti(
        mag_diffs,
        mc=dmc,
        delta_m=delta_m,
        b_parameter=b_parameter,
        return_std=return_std,
    )

    if return_n:
        if type(out) is tuple:
            return out + tuple([len(mag_diffs)])
        else:
            return out, len(mag_diffs)
    else:
        return out


def make_more_incomplete(
    magnitudes: np.ndarray,
    times: dt.datetime,
    delta_t: np.timedelta64 = np.timedelta64(60, "s"),
    return_idx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered magnitudes and times. Filter the magnitudes and times in
    the following way: If an earthquake is smaller than the previous one and
    less than ``delta_t`` away, the earthquake is removed.

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
-       Earth, 129(2):e2023JB027849, 2024.

    Args:
        magnitudes: array of magnitudes, sorted in time (first
                entry is the earliest earthquake).
        times:      array of datetime objects of occurrence of each earthquake
        delta_t:    time window in seconds to filter out events. default is 60
                seconds.
        return_idx: if True the indices of the events that were removed are
                returned

    Returns:
        magnitudes: filtered array of magnitudes
        times:      filtered array of datetime objects
        idx:        indices of the events that were removed

    """

    # sort magnitudes in time
    idx_sort = np.argsort(times)
    magnitudes = magnitudes[idx_sort]
    times = times[idx_sort]

    idx_del = []
    for ii in range(1, len(magnitudes)):
        # get all times that are closer than delta_t
        idx_close = np.where(times[ii] - times[:ii] < delta_t)[0]

        # check if these events are larger than the current event
        idx_del_loop = np.where(magnitudes[idx_close] > magnitudes[ii])[0]

        # if there are any, remove the current event
        if len(idx_del_loop) > 0:
            idx_del.append(ii)

    magnitudes = np.delete(magnitudes, idx_del)
    times = np.delete(times, idx_del)

    if return_idx is True:
        return magnitudes, times, idx_del

    return magnitudes, times


def estimate_b_laplace(
    magnitudes: np.ndarray,
    delta_m: float = 0,
    b_parameter: str = "b_value",
    return_std: bool = False,
    return_n: bool = False,
) -> float | tuple[float, float]:
    """Return the b-value estimate calculated using
    all the  differences between magnitudes.
    (This has a little less variance than the
    :func:`seismostats.analysis.estimate_b_positive`
    method.)

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

    out = estimate_b_tinti(
        mag_diffs,
        mc=delta_m,
        delta_m=delta_m,
        b_parameter=b_parameter,
        return_std=return_std,
    )

    if return_n:
        if type(out) is tuple:
            return out + tuple([len(mag_diffs)])
        else:
            return out, len(mag_diffs)
    else:
        return out


def estimate_b_weichert(
    magnitudes: np.ndarray,
    dates: list[np.datetime64],
    completeness_table: np.ndarray,
    mag_max: int | float,
    last_year: int | float | None = None,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> tuple[float, float, float, float, float]:
    """Return the b-value estimate calculated using the
    Weichert (1980) algorithm, for the case of unequal
    completeness periods for different magnitude values.

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

            >>> np.array([[ 3.95, 1980],
            ...          [ 4.95, 1920],
            ...          [ 5.95, 1810],
            ...          [ 6.95, 1520]])

        mag_max: maximum possible magnitude
        last_year: last year of observation (the default is None, in which case
              it is set to the latest year in years).
        delta_m: magnitude resolution, the default is 0.1.
        b_parameter:either 'b-value', then the corresponding value of the
                    Gutenberg-Richter law is returned, otherwise 'beta'
                    from the exponential distribution [p(M) = exp(-beta*(M-mc))]

    Returns:
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


def estimate_b_kijko_smit(
    magnitudes: np.ndarray,
    dates: list[np.datetime64],
    completeness_table: np.ndarray,
    last_year: int | float | None = None,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> tuple[float, float, float, float]:
    """Return the b-value estimate calculated using the Kijko and Smit (2012)
    algorithm for the case of unequal completeness periods for different
    magnitude values.

    Source:
        Kijko, A. and Smit, A., 2012. Extension of the Aki-Utsu b-value
        estimator for incomplete catalogs. Bulletin of the Seismological
        Society of America, 102(3), pp.1283-1287.


    Args:
        magnitudes: vector of earthquake magnitudes
        dates: list of datetime objects of occurrence of each earthquake
        completeness_table: Nx2 array, where the first column
            contains the leftmost edge of magnitude bins and
            the second column the associated year of completeness, i.e.
            the year after which all earthquakes larger than the value in
            the first column are considered detected. An example is given
            below:

            >>> np.array([[ 3.95, 1980],
            ...           [ 4.95, 1920],
            ...           [ 5.95, 1810],
            ...           [ 6.95, 1520]])

        last_year: last year of observation (the default is None, in which case
              it is set to the latest year in years).
        delta_m: magnitude resolution, the default is 0.1.
        b_parameter:either 'b-value', then the corresponding value of the
                    Gutenberg-Richter law is returned, otherwise 'beta'
                    from the exponential distribution [p(M) = exp(-beta*(M-mc))]

    Returns:
        b_parameter: maximum likelihood point estimate of 'b-value' or 'beta'
        std_b_parameter: standard error of b_parameter
        rate_at_lmc: maximum likelihood point estimate of earthquake rate
                     at the lower magnitude of completeness
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
        in np.arange(
            completeness_table[0, 0],
            np.max(magnitudes) + delta_m + 0.001,
            delta_m,
        )
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
    years = np.array(dates).astype("datetime64[Y]").astype(int) + 1970

    # get last year of catalogue if last_year not defined
    last_year = last_year if last_year else np.max(years)

    # Get the magnitudes and completeness years as separate arrays
    completeness_magnitudes = completeness_table[:, 0]
    completeness_years = completeness_table[:, 1]

    # Obtain the completeness magnitudes for each value in years
    insertion_indices = np.searchsorted(-completeness_years, -years)

    # get complete sub-catalogues and number of complete events in each of them
    sub_catalogues = []
    ncomplete = 0
    for idx in range(len(completeness_magnitudes)):
        sub_catalogues.append(
            magnitudes[
                (insertion_indices == idx)
                & (magnitudes > completeness_magnitudes[idx])
            ]
        )
        ncomplete += len(sub_catalogues[-1])

    # Equation (7) from Kijko, Smit (2012)
    estimator_terms = []
    for ii, subcat_magnitudes in enumerate(sub_catalogues):
        # get sub-catalogue beta-value as per Aki-Utsu
        sub_beta = 1 / (
            np.average(subcat_magnitudes) - completeness_magnitudes[ii]
        )
        estimator_terms.append((len(subcat_magnitudes) / ncomplete) / sub_beta)
    beta = 1 / np.sum(estimator_terms)

    # standard deviation of b/beta (Equation 8)
    std_b = beta / np.sqrt(ncomplete)

    if b_parameter == "b_value":
        b = beta_to_b_value(beta)
        std_b = beta_to_b_value(std_b)

    # get rate assuming Poisson process (Equation 10)
    denominator_rate = 0
    for idx in range(len(completeness_magnitudes)):
        denominator_rate += (last_year - completeness_years[idx]) * np.exp(
            (-beta)
            * (completeness_magnitudes[idx] - completeness_magnitudes[0])
        )

    rate_at_lmc = ncomplete / denominator_rate
    a_val = np.log10(rate_at_lmc) + (beta / np.log(10)) * (
        completeness_magnitudes[0] + delta_m * 0.5
    )

    return b, std_b, rate_at_lmc, a_val


def shi_bolt_confidence(
    magnitudes: np.ndarray,
    b: float | None = None,
    b_parameter: str = "b_value",
) -> float:
    """Return the Shi and Bolt (1982) confidence limit of the b-value or
    beta.

    Source:
        Shi and Bolt, BSSA, Vol. 72, No. 5, pp. 1677-1687, October 1982

    Args:
        magnitudes: numpy array of magnitudes
        b:          known or estimated b-value/beta of the magnitudes
        b_parameter:either either 'b_value' or 'beta'

    Returns:
        std_b:  confidence limit of the b-value/beta value (depending on input)
    """
    # standard deviation in Shi and Bolt is calculated with 1/(N*(N-1)), which
    # is by a factor of sqrt(N) different to the std(x, ddof=1) estimator
    assert (
        b_parameter == "b_value" or b_parameter == "beta"
    ), "please choose either 'b_value' or 'beta' as b_parameter"

    std_b = (
        np.log(10) * b**2 * np.std(magnitudes) / np.sqrt(len(magnitudes) - 1)
    )
    if b_parameter == "beta":
        std_b = (std_b) / np.log(10)

    return std_b
