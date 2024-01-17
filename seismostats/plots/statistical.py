# standard
import matplotlib.pyplot as plt
import numpy as np

# statistical
from scipy.stats import norm

# Own functions
from seismostats.analysis.estimate_beta import (
    estimate_b_positive,
    estimate_b_tinti,
)


def plot_mc_vs_b(
    magnitudes: np.ndarray,
    mcs: np.ndarray,
    delta_m: float = 0.1,
    method: str = "tinti",
    confidence_intvl: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
) -> plt.Axes:
    """Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes: magnitudes of the catalog
        mcs:        completeness magnitudes (list or numpy array)
        delta_m:    discretization of the magnitudes
        method:     method used for b-value estimation, either 'tinti' or
                    'positive' or 'positive_postcut'. positive_postcut is the
                    same as 'positive' but with the postcut method (differences
                    are taken before cutting the magnitudes below the
                    completeness magnitude). The mcs are then interpreted as
                    dmcs.
        confidence_intvl:   confidence interval that should be plotted
        ax:         axis where figure should be plotted
        color:      color of the data

    Returns:
        ax that was plotted on
    """

    # try except
    try:
        if method == "tinti":
            results = [
                estimate_b_tinti(
                    magnitudes[magnitudes >= mc],
                    mc,
                    delta_m=delta_m,
                    error=True,
                )
                for mc in mcs
            ]
        elif method == "positive":
            results = [
                estimate_b_positive(
                    magnitudes[magnitudes >= mc],
                    delta_m=delta_m,
                    error=True,
                )
                for mc in mcs
            ]
        elif method == "positive_postcut":
            mag_diffs = np.diff(magnitudes)
            mag_diffs = mag_diffs[mag_diffs > 0]
            results = [
                estimate_b_tinti(
                    mag_diffs[mag_diffs >= mc],
                    mc=mc,
                    delta_m=delta_m,
                    error=True,
                )
                for mc in mcs
            ]
        else:
            raise ValueError(
                "Method must be either 'tinti', 'positive' or"
                "'positive_postcut'"
            )

        b_values, b_errors = zip(*results)
        b_values = np.array(b_values)
        b_errors = np.array(b_errors)
    except ValueError as err:
        print(err)
        return

    if ax is None:
        fig, ax = plt.subplots()

    # Plotting
    error_factor = norm.ppf((1 + confidence_intvl) / 2)
    ax.plot(mcs, b_values, "-o", color=color)
    ax.fill_between(
        mcs,
        b_values - error_factor * b_errors,
        b_values + error_factor * b_errors,
        alpha=0.2,
        color=color,
    )
    ax.set_xlabel("Completeness magnitude $m_c$")
    ax.set_ylabel("b-value")
    ax.grid(True)

    return ax
