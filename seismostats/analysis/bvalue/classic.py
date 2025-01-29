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
        return _mle_estimator(self.magnitudes,
                              mc=self.mc,
                              delta_m=self.delta_m,
                              weights=self.weights)
