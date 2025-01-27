import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option


class BPositiveBValueEstimator(BValueEstimator):
    '''
    B-value estimator using positive differences between
    consecutive magnitudes.

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126,
        Issue 2)
    '''

    weights_supported = True

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  weights: np.ndarray | None = None,
                  dmc: float | None = None) -> float:
        '''
        Return the b-value estimate calculated using the
        positive differences between consecutive magnitudes.

        Args:
            magnitudes: Array of magnitudes
            mc:         Completeness magnitude
            delta_m:    Discretization of magnitudes.
            weights:    Array of weights for the magnitudes.
            dmc:        Cutoff value for the differences (differences
                        below this value are not considered). If None,
                        the cutoff is set to delta_m.
        '''

        return super().calculate(magnitudes,
                                 mc=mc,
                                 delta_m=delta_m,
                                 weights=weights,
                                 dmc=dmc)

    def _estimate(self, dmc: float | None = None) -> float:

        self.dmc = dmc or self.delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')

        elif self.dmc < self.delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

        # only take the values where the next earthquake is d_mc larger than the
        # previous one. delta_m is added to avoid numerical errors
        self.magnitudes = np.diff(self.magnitudes)
        is_larger = self.magnitudes > self.dmc - self.delta_m / 2
        self.magnitudes = abs(self.magnitudes[is_larger])

        if self.weights is not None:
            # use weight of second earthquake of a difference
            self.weights = self.weights[1:][is_larger]

        classic_estimator = ClassicBValueEstimator()

        return classic_estimator.calculate(self.magnitudes,
                                           mc=self.dmc,
                                           delta_m=self.delta_m,
                                           weights=self.weights)
