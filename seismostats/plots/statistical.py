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
    delta_m: float,
    b_method: BValueEstimator = ClassicBValueEstimator,
    confidence_intvl: float = 0.95,
    ax: plt.Axes | None = None,
    color: str = "blue",
    label: str = None,
    **kwargs,
) -> plt.Axes:
    """Plots the estimated b-value in dependence of the completeness magnitude.

    Args:
        magnitudes: magnitudes of the catalog
        mcs:        completeness magnitudes (list or numpy array)
        delta_m:    discretization of the magnitudes
        method:     method used for b-value estimation
        confidence_intvl:   confidence interval that should be plotted
        ax:         axis where figure should be plotted
        color:      color of the data
        **kwargs:   Additional keyword arguments for the b-value
                estimator.

    Returns:
        ax that was plotted on
    """

    b_values = []
    b_errors = []
    estimator = b_method()

    for mc in mcs:
        estimator.calculate(
            magnitudes, mc=mc, delta_m=delta_m, **kwargs)
        b_values.append(estimator.b_value)
        b_errors.append(estimator.std)

    b_values = np.array(b_values)
    b_errors = np.array(b_errors)

    if ax is None:
        _, ax = plt.subplots()

    error_factor = norm.ppf((1 + confidence_intvl) / 2)
    ax.plot(mcs, b_values, "-o", color=color, label=label)
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
    if label is not None:
        ax.legend()

    return ax
