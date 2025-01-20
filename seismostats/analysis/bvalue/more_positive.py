import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
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

        if self.weights is not None:
            weights_pos = np.ones_like(mag_diffs)

        for ii in range(len(self.magnitudes) - 1):
            for jj in range(ii + 1, len(self.magnitudes)):
                mag_diff_loop = self.magnitudes[jj] - self.magnitudes[ii]
                if mag_diff_loop > self.dmc - self.delta_m / 2:
                    mag_diffs[ii] = mag_diff_loop
                    if self.weights is not None:
                        # use weight of second earthquake of a difference
                        weights_pos[ii] = self.weights[jj]
                    break

        # only take the values where the next earthquake is larger
        is_larger = mag_diffs > -self.delta_m / 2

        if self.weights is not None:
            self.weights = weights_pos[is_larger]

        self.magnitudes = abs(mag_diffs[is_larger])

        classic_estimator = ClassicBValueEstimator()

        return classic_estimator.calculate(self.magnitudes,
                                           mc=self.dmc,
                                           delta_m=self.delta_m,
                                           weights=self.weights)
