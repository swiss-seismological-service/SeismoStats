import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value


class ClassicBValueEstimator(BValueEstimator):
    '''
    Estimator for the b-value using the maximum likelihood estimator.

    Source:
        - Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
        - Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
        America, 77(6), 2125-2134.)
    '''

    weights_supported = True

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:

        if self.delta_m > 0:
            p = 1 + self.delta_m / \
                np.average(self.magnitudes - self.mc, weights=self.weights)
            beta = 1 / self.delta_m * np.log(p)
        else:
            beta = 1 / np.average(self.magnitudes
                                  - self.mc, weights=self.weights)

        return beta_to_b_value(beta)
