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
            >>> from seismostats.analysis import ClassicAValueEstimator

            >>> magnitudes = np.array([2.1, 2.3, 2.0, 2.0, 2.1, 2.2, 2.1, 2.3,
            ...                        2.0, 2.0])

            >>> my_estimator = ClassicAValueEstimator()
            >>> my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1)

            >>> my_estimator.a_value

            1.0
    '''

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        return np.log10(len(self.magnitudes))
