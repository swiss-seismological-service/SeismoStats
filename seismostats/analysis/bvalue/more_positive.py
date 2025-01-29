import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.analysis.bvalue.utils import find_next_larger
from seismostats.utils._config import get_option


class BMorePositiveBValueEstimator(BValueEstimator):
    """
    B-value estimator using the next positive differences (this means that
    almost every magnitude has a difference, as opposed to the b-positive
    method which results in half the data).

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    """

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

    def _estimate(self, dmc: float) -> float:

        self.dmc = dmc or self.delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')
        elif self.dmc < self.delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

        mag_diffs = -np.ones(len(self.magnitudes) - 1) * self.delta_m

        idx_next_larger = find_next_larger(
            self.magnitudes, self.delta_m, self.dmc)
        mag_diffs = self.magnitudes[idx_next_larger] - self.magnitudes

        idx_no_next = idx_next_larger == 0
        self.magnitudes = mag_diffs[~idx_no_next]

        if self.weights is not None:
            weights = self.weights[idx_next_larger]
            self.weights = weights[~idx_no_next]

        return _mle_estimator(self.magnitudes,
                              mc=self.mc,
                              delta_m=self.delta_m,
                              weights=self.weights)
