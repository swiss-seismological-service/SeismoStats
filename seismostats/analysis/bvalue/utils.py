from typing import Literal

import numpy as np


def beta_to_b_value(beta: float) -> float:
    '''converts the beta value to the b-value  of the Gutenberg-Richter law

    Args:
        beta: beta value

    Returns:
        b_value: corresponding b-value
    '''
    return beta / np.log(10)


def b_value_to_beta(b_value: float) -> float:
    '''converts the b-value to the beta value of the exponential distribution

    Args:
        b_value: b-value

    Returns:
        beta: corresponding beta value
    '''
    return b_value * np.log(10)


def shi_bolt_confidence(
    magnitudes: np.ndarray,
    b: float,
    weights: np.ndarray | None = None,
    b_parameter: Literal['b_value', 'beta'] = 'b_value'
) -> float:
    '''Return the Shi and Bolt (1982) confidence limit of the b-value or
    beta.

    Source:
        Shi and Bolt, BSSA, Vol. 72, No. 5, pp. 1677-1687, October 1982

    Args:
        magnitudes: numpy array of magnitudes
        weights:    numpy array of weights for each magnitude
        b:          known or estimated b-value/beta of the magnitudes
        b_parameter:either either 'b_value' or 'beta'

    Returns:
        std_b:  confidence limit of the b-value/beta value (depending on input)
    '''
    # standard deviation in Shi and Bolt is calculated with 1/(N*(N-1)), which
    # is by a factor of sqrt(N) different to the std(x, ddof=1) estimator
    if b_parameter not in ['b_value', 'beta']:
        raise ValueError('b_parameter must be either "b_value" or "beta"')

    std_mags = np.sqrt(np.average(np.square(
        magnitudes - np.average(magnitudes, weights=weights)), weights=weights))
    if weights is None:
        len_mags = len(magnitudes)
    else:
        len_mags = np.sum(weights)

    std_b = (
        np.log(10) * b**2 * std_mags / np.sqrt(len_mags - 1)
    )
    if b_parameter == 'beta':
        std_b = (std_b) / np.log(10)

    return std_b


def find_next_larger(magnitudes: np.array,
                     delta_m: float,
                     dmc: float | None):
    """Takes an array of magnitudes and returns an array of indices of the next
    largest event for each element. For each magnitude[ii], magnitude[idx[ii]]
    is the next larger event in the series. Example: magnitudes = [10, 4, 3, 9]
    result in [0, 3, 3, 0]. Note that the value of idx is 0 if no
    next magnitude exists.

    Args:
        magnitudes:     ordered magnitudes (in the dimension of interest, e.g.
                    time)
        delta_m:        discretization of the magnitudes.
        dmc:            minimum magnitude difference between consecutive events.
                    If None, the default value is delta_m.

    """
    if dmc is None:
        dmc = delta_m

    idx_next_larger = np.zeros(len(magnitudes))
    for ii in range(len(magnitudes) - 1):
        for jj in range(ii + 1, len(magnitudes)):
            mag_diff_loop = magnitudes[jj] - magnitudes[ii]
            if mag_diff_loop > dmc - delta_m / 2:
                idx_next_larger[ii] = jj
                break
    return idx_next_larger.astype(int)


def make_more_incomplete(
    magnitudes: np.ndarray,
    times: np.array,
    delta_t: np.timedelta64 = np.timedelta64(60, 's'),
    return_idx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    '''Return filtered magnitudes and times. Filter the magnitudes and times in
    the following way: If an earthquake is smaller than the previous one and
    less than ``delta_t`` away, the earthquake is removed.

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Args:
        magnitudes: array of magnitudes, sorted in time (first
                entry is the earliest earthquake).
        times:      array of datetime objects of occurrence of each earthquake
        delta_t:    time window in seconds to filter out events. default is 60
                seconds.
        return_idx: if True the indices of the events that were kept are
                returned

    Returns:
        magnitudes: filtered array of magnitudes
        times:      filtered array of datetime objects
        idx:        indices of the events that were kept
        '''

    # sort magnitudes in time
    idx_sort = np.argsort(times)
    magnitudes = magnitudes[idx_sort]
    times = times[idx_sort]

    idx = np.full(len(magnitudes), True)
    for ii in range(1, len(magnitudes)):
        # get all times that are closer than delta_t
        idx_close = np.where(times[ii] - times[:ii] < delta_t)[0]

        # check if these events are larger than the current event
        idx_loop = magnitudes[idx_close] > magnitudes[ii]

        # if there are any, remove the current event
        if sum(idx_loop) > 0:
            idx[ii] = False

    magnitudes = magnitudes[idx]
    times = times[idx]

    if return_idx is True:
        return magnitudes, times, idx

    return magnitudes, times
