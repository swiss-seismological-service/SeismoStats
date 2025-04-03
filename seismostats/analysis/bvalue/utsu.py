import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value


class UtsuBValueEstimator(BValueEstimator):
    '''
    Estimator to calculate the b-value and other parameters of the
    Gutenberg-Richter (GR) law using the Utsu method.

    Source:
        Utsu 1965 (Geophysical bulletin of the Hokkaido University,
        vol 13, pp 99-103).

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis import UtsuBValueEstimator

            >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3,
            ...                        2.7, 2.2, 2.4, 2. , 2.7, 2.2, 2.3, 2.1,
            ...                        2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5])

            >>> my_estimator = UtsuBValueEstimator()
            >>> my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1)

            >>> my_estimator.b_value

            1.1088369750721319
    '''

    _weights_supported = True

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        beta = 1 / np.average(self.magnitudes - self.mc
                              + self.delta_m / 2, weights=self.weights)
        return beta_to_b_value(beta)
