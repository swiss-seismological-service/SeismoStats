import numpy as np

from seismostats.analysis.avalue.base import AValueEstimator


class ClassicAValueEstimator(AValueEstimator):
    '''
    Returns the a-value of the Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \\cdot (m - m_{ref})},

    where :math:`N(m)` is the number of events with magnitude larger than
    or equal to :math:`m` that occurred in the timeframe of the catalog,
    :math:`a` and :math:`b` are the a- and b-value, and :math:`m_{ref}`
    is the reference magnitude above which earthquakes are counted.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis.avalue import ClassicAValueEstimator

            >>> magnitudes = np.array([2.1, 2.3, 2.0, 2.0, 2.1, 2.2, 2.1, 2.3,
            ...                        2.0, 2.0])
            >>> mc = 2.0
            >>> delta_m = 0.1

            >>> my_estimator = ClassicAValueEstimator()
            >>> a_value = my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=mc, delta_m=delta_m)

            >>> print(a_value)

            1.0

        .. code-block:: python

            >>> print("used magnitudes:", my_estimator.magnitudes)
            >>> print("used mc:        ", my_estimator.mc)
            >>> print("used delta_m:   ", my_estimator.delta_m)
            >>> print("a-value:        ", my_estimator.a_value)

            used magnitudes: [2.1 2.3 2.  2.  2.1 2.2 2.1 2.3 2.  2. ]
            used mc:         2.0
            used delta_m:    0.1
            a-value:         1.0
    '''

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        return np.log10(len(self.magnitudes))
