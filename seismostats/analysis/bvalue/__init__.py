# flake8: noqa
import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.bvalue.more_positive import \
    BMorePositiveBValueEstimator
from seismostats.analysis.bvalue.positive import BPositiveBValueEstimator
from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               beta_to_b_value,
                                               shi_bolt_confidence)
from seismostats.analysis.bvalue.utsu import UtsuBValueEstimator


def estimate_b(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = 'b_value',
    return_std: bool = False,
    method: BValueEstimator = ClassicBValueEstimator,
    return_n: bool = False,
    *args,
    **kwargs
) -> float | tuple[float, float] | tuple[float, float, float]:

    estimator = method(mc=mc, delta_m=delta_m, *args, **kwargs)

    if b_parameter == 'b_value':
        b = estimator(magnitudes, weights=weights)
    elif b_parameter == 'beta':
        b = estimator.estimate_beta(magnitudes, weights=weights)
    else:
        raise ValueError('b_parameter must be either "b_value" or "beta"')

    out = b

    if return_std:
        out = (out, estimator.std)

    if return_n:
        out = (*tuple(out), estimator.n)

    return out
