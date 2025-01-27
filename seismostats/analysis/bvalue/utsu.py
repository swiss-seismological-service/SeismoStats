import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value


class UtsuBValueEstimator(BValueEstimator):
    '''
    Estimator for the b-value using the maximum likelihood estimator.

    Source:
        Utsu 1965 (Geophysical bulletin of the Hokkaido University,
        vol 13, pp 99-103).
    '''

    weights_supported = True

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        beta = 1 / np.average(self.magnitudes - self.mc
                              + self.delta_m / 2, weights=self.weights)
        return beta_to_b_value(beta)
