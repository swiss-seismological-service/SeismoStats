import warnings

import numpy as np

from seismostats.analysis.b_values.base import BValueEstimator
from seismostats.analysis.b_values.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option


class BPositiveEstimator(BValueEstimator):
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

    weights_supported = False

    def __init__(self, dmc: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dmc = dmc or self.delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')
        elif self.dmc < self.delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

    def _estimate(self):
        self.magnitudes = np.diff(self.magnitudes)
        # only take the values where the next earthquake is d_mc larger than the
        # previous one. delta_m is added to avoid numerical errors
        self.magnitudes = abs(
            self.magnitudes[self.magnitudes > self.dmc - self.delta_m / 2])

        classic_estimator = ClassicBValueEstimator(mc=self.mc,
                                                   delta_m=self.delta_m)

        return classic_estimator(self.magnitudes)
