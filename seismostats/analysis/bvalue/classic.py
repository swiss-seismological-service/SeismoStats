import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value


def _mle_estimator(magnitudes: np.ndarray,
                   mc: float,
                   delta_m: float,
                   weights: np.ndarray | None = None) -> float:
    '''
    Internal function for the classic b-value estimator. For b-value
    estimation use `ClassicBValueEstimator` instead.
    '''
    if delta_m > 0:
        p = 1 + delta_m / np.average(magnitudes - mc, weights=weights)
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average(magnitudes - mc, weights=weights)

    return beta_to_b_value(beta)


class ClassicBValueEstimator(BValueEstimator):
    '''
    Estimator to calculate the b-value and other parameters of the
    Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \\cdot (m - m_{ref})},

    where :math:`N(m)` is the number of events with magnitude larger than
    or equal to :math:`m` that occurred in the timeframe of the catalog,
    :math:`a` and :math:`b` are the a- and b-value, and :math:`m_{ref}`
    is the reference magnitude above which earthquakes are counted.

    Source:
        - Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        - Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
          America, 77(6), 2125-2134.)

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis import ClassicBValueEstimator

            >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3,
            ...                        2.7, 2.2, 2.4, 2. , 2.7, 2.2, 2.3, 2.1,
            ...                        2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5])

            >>> my_estimator = ClassicBValueEstimator()
            >>> my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1)

            >>> my_estimator.b_value

            1.114920128810535
    '''

    _weights_supported = True

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        return _mle_estimator(self.magnitudes,
                              mc=self.mc,
                              delta_m=self.delta_m,
                              weights=self.weights)
