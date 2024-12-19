# standard
import matplotlib.pyplot as plt
import numpy as np
# statistical
from scipy.stats import norm

# Own functions
from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator


def plot_mc_vs_b(
    magnitudes: np.ndarray,
    mcs: np.ndarray,
    dmc: float | None = None,
    delta_m: float = 0.1,
    b_method: BValueEstimator = ClassicBValueEstimator,
    confidence_intvl: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
) -> plt.Axes:
    """Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes: magnitudes of the catalog
        mcs:        completeness magnitudes (list or numpy array)
        dmc:        if a positive b-value estimator is used, this is the
                minimum difference that is considered. For the
                ClassicBValueEstimator, leave this as the default None.
        delta_m:    discretization of the magnitudes
        method:     method used for b-value estimation
        confidence_intvl:   confidence interval that should be plotted
        ax:         axis where figure should be plotted
        color:      color of the data

    Returns:
        ax that was plotted on
    """

    b_values = []
    b_errors = []

    if dmc is None:
        for mc in mcs:
            estimator = b_method(mc=mc, delta_m=delta_m)
            b_values.append(estimator(magnitudes))
            b_errors.append(estimator.std)
    else:
        for mc in mcs:
            estimator = b_method(mc=mc, delta_m=delta_m, dmc=dmc)
            b_values.append(estimator(magnitudes))
            b_errors.append(estimator.std)

    b_values = np.array(b_values)
    b_errors = np.array(b_errors)

    if ax is None:
        _, ax = plt.subplots()

    # Plotting: this either
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
