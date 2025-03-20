
import warnings

import numpy as np

from seismostats.analysis.bvalue.utils import beta_to_b_value
from seismostats.utils._config import get_option


def estimate_b_kijko_smit(
    magnitudes: np.ndarray,
    times: list[np.datetime64],
    completeness_table: np.ndarray,
    last_year: int | float | None = None,
    delta_m: float = 0.1,
    b_parameter: str = "b_value",
) -> tuple[float, float, float, float]:
    """
    Returns the b-value of the Gutenberg-Richter (GR) law
    calculated using the Kijko and Smit (2012) algorithm
    for the case of unequal completeness periods for different
    magnitude values.

    Source:
        Kijko, A. and Smit, A., 2012. Extension of the Aki-Utsu b-value
        estimator for incomplete catalogs. Bulletin of the Seismological
        Society of America, 102(3), pp.1283-1287.


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
            ...           [ 4.95, 1920],
            ...           [ 5.95, 1810],
            ...           [ 6.95, 1520]])

        last_year:          Last year of observation (the default is `None`,
            in which case it is set to the latest year in years).
        delta_m:            Bin size of discretized magnitudes, default is 0.1.
        b_parameter:        either 'b-value', then the corresponding value of
            the Gutenberg-Richter law is returned, otherwise 'beta'
            from the exponential distribution [p(M) = exp(-beta*(M-mc))]

    Returns:
        b_parameter:        Maximum likelihood estimate of 'b-value' or 'beta'.
        std_b_parameter:    Standard error of b_parameter.
        rate_at_lmc:        Maximum likelihood point estimate of earthquake rate
                    at the lower magnitude of completeness.
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
    years = np.array(times).astype("datetime64[Y]").astype(int) + 1970

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
