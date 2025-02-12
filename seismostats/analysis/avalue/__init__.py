# flake8: noqa

import numpy as np

from seismostats.analysis.avalue.base import AValueEstimator
from seismostats.analysis.avalue.classic import ClassicAValueEstimator
from seismostats.analysis.avalue.more_positive import \
    AMorePositiveAValueEstimator
from seismostats.analysis.avalue.positive import APositiveAValueEstimator


def estimate_a(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float,
    scaling_factor: float | None = None,
    m_ref: float | None = None,
    b_value: float | None = None,
    method: AValueEstimator = ClassicAValueEstimator,
    *args,
    **kwargs
) -> float:

    estimator = method()
    estimator.calculate(magnitudes, mc=mc, delta_m=delta_m,
                        scaling_factor=scaling_factor, m_ref=m_ref,
                        b_value=b_value, *args, **kwargs)

    return estimator.a_value
