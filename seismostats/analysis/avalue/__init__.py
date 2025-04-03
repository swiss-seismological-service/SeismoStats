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
    **kwargs
) -> float:
    '''
    Returns the a-value of the Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \\cdot (m - m_{ref})},

    where :math:`N(m)` is the number of events with magnitude larger than 
    or equal to :math:`m` that occurred in the timeframe of the catalog,
    :math:`a` and :math:`b` are the a- and b-value, and :math:`m_{ref}`
    is the reference magnitude above which earthquakes are counted.

    Args:
        magnitudes:     Array of magnitudes.
        mc:             Completeness magnitude.
        delta_m:        Bin size of discretized magnitudes.
        scaling_factor: Scaling factor.
                    If given, this is used to normalize the number of
                    observed events. For example: Volume or area of the
                    region considered or length of the time interval,
                    given in the unit of interest.
        m_ref:          Reference magnitude for which the a-value
                    is estimated.
        b_value:        b-value of the Gutenberg-Richter law. Only relevant
                    when `m_ref` is not `None`.
        method:         AValueEstimator class to use for calculation.
        **kwargs:       Additional parameters to be passed to the :func:`calculate`
                    method.

    Returns:
        a: a-value of the Gutenberg-Richter law.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis import estimate_a

            >>> magnitudes = np.array([2.1, 2.3, 2.0, 2.0, 2.1, 2.2, 2.1, 2.3, 2.0, 2.0])
            >>> mc = 2.0
            >>> delta_m = 0.1
            >>> a = estimate_a(magnitudes, mc, delta_m)
            >>> a

            1.0

        .. code-block:: python

            >>> from seismostats.analysis import APositiveAValueEstimator

            >>> times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> a = estimate_a(magnitudes, mc, delta_m, times=times, method=APositiveAValueEstimator)
            >>> a

            0.9542425094393249
    '''

    estimator = method()
    estimator.calculate(magnitudes, mc=mc, delta_m=delta_m,
                        scaling_factor=scaling_factor, m_ref=m_ref,
                        b_value=b_value, **kwargs)

    return estimator.a_value
