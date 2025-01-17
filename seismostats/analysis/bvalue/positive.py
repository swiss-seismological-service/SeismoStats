import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option


class BPositiveBValueEstimator(BValueEstimator):
    '''Return the b-value estimate calculated using the
    positive differences between consecutive magnitudes.

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        delta_m:    discretization of magnitudes. default is no discretization.
        dmc:       cutoff value for the differences (diffferences below this
                value are not considered). If None, the cutoff is set to delta_m
    '''

    weights_supported = True

    def __init__(self, *args, dmc: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.dmc: float
        self._register_attribute('dmc', dmc or self.delta_m)

    def _estimate(self,
                  magnitudes: np.ndarray,
                  weights: np.ndarray | None
                  ) -> tuple[float, np.ndarray, np.ndarray | None]:

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')

        elif self.dmc < self.delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

        # only take the values where the next earthquake is d_mc larger than the
        # previous one. delta_m is added to avoid numerical errors
        magnitudes = np.diff(magnitudes)
        is_larger = magnitudes > self.dmc - self.delta_m / 2
        magnitudes = abs(magnitudes[is_larger])

        if weights is not None:
            # use weight of second earthquake of a difference
            weights = weights[1:][is_larger]

        classic_estimator = ClassicBValueEstimator(magnitudes,
                                                   mc=self.dmc,
                                                   delta_m=self.delta_m,
                                                   weights=weights)

        return classic_estimator.b_value(), magnitudes, weights
