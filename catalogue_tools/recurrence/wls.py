import math

import numpy as np
from numpy.typing import ArrayLike


def weighted_least_squares(years: ArrayLike,
                           magnitudes: ArrayLike,
                           completeness_table: ArrayLike) -> tuple:
    """
    Calculates the a and b values of a catalogue using weighted last squares.

    :param years:               The years when the earthquakes occurred.
    :param magnitudes:          Magnitudes of the earthquakes.
    :param completeness_table:  Table with years and respective Mc's for the
                                catalogue provided.

    :returns: Tuple with (a value, std of a, b value, std of b)
    """

    # end year of catalogue
    end_year = years.max()

    bin_width = 0.5
    minmag = math.floor(magnitudes.min())
    maxmag = math.ceil(magnitudes.max())

    # create the bins
    bins = np.arange(minmag, maxmag, bin_width)

    # prepare arrays for results
    nbins = len(bins)
    n_mags = np.zeros(nbins)
    rates = np.zeros_like(n_mags)
    midpoints = np.zeros_like(n_mags)
    duration = np.zeros_like(n_mags)
    start_years = np.zeros_like(n_mags)

    # Loop through each bin
    for i, row in enumerate(bins):
        # Starting year is the completeness year
        start_years[i] = \
            completeness_table[:, 0][completeness_table[:, 1] <= row].min()

        # Duration is the time between the end of the catalogue and
        # and the year of completeness
        duration[i] = end_year - start_years[i] + 1

        # select the years >= the completeness year
        selected_years = years >= start_years[i]

        # Define the upper and lower magnitude bounds of the bin
        mlow = row
        mhigh = row + bin_width
        midpoints[i] = (mlow + mhigh) / 2.

        # find earthquakes within the magnitude range
        selected_magnitudes = np.logical_and(
            magnitudes[selected_years] >= mlow,
            magnitudes[selected_years] < mhigh
        )
        # Count the number of earthquakes selected
        n_mags[i] = np.sum(selected_magnitudes)
        # Divide by the duration of completeness for the magnitude bin
        rates[i] = float(n_mags[i]) / duration[i]

    # Get the cumulative rates
    cum_rates = np.zeros_like(n_mags)
    for i in range(nbins):
        # Sum the rates greater than or equal to each bin
        cum_rates[i] = np.sum(rates[i:])

    # the best fitting parameters and the covariance matrix
    best_fit, C = np.polyfit(midpoints,  # x-values
                             # y-values, use the log10 of the cumulative rates
                             np.log10(cum_rates),
                             # order of polynomial
                             deg=1,
                             cov=True)  # Also return the covariance matrix

    # The variance of the best fitting parameters can be found on the
    # leading diagonal of the covarince matrix. Take the square root of
    # these to find the standard deviations
    uncertainties = np.sqrt(np.diag(C))
    # Best-fit is an array with the [b-value, a-value]
    bls, als = best_fit
    # in case b-value is negative turn it positive
    bls = np.fabs(bls)

    return (als, uncertainties[1], bls, uncertainties[0])
