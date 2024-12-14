import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option


class BMorePositiveBValueEstimator(BValueEstimator):
    """Return the b-value estimate calculated using the
    next positive differences (this means that almost every magnitude has a
    difference, as opposed to the b-positive method which results in half the
    data).

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Args:
        delta_m:    discretization of magnitudes. default is no discretization.
        dmc:        cutoff value for the differences (differences below this
                    value are not considered). If None, it is set to delta_m.
    """

    weights_supported = True

    def __init__(self, dmc: float | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dmc = dmc or self.delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')
        elif self.dmc < self.delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

    def _estimate(self):

        mag_diffs = - np.ones(len(self.magnitudes) - 1) * self.delta_m
        if self.weights is not None:
            # weight vector of same length as mag diffs
            weights = - np.ones(len(self.magnitudes) - 1) * self.delta_m
        for ii in range(len(self.magnitudes) - 1):
            for jj in range(ii + 1, len(self.magnitudes)):
                mag_diff_loop = self.magnitudes[jj] - self.magnitudes[ii]
                if mag_diff_loop > self.dmc - self.delta_m / 2:
                    mag_diffs[ii] = mag_diff_loop
                    if self.weights is not None:
                        # use weight of second earthquake of a difference
                        weights[ii] = self.weights[jj]
                    break

        # only take the values where the next earthquake is larger
        if self.weights is not None:
            self.weights = weights[mag_diffs > - self.delta_m / 2]
        self.magnitudes = abs(mag_diffs[mag_diffs > - self.delta_m / 2])

        classic_estimator = ClassicBValueEstimator(mc=self.dmc,
                                                   delta_m=self.delta_m)

        return classic_estimator(self.magnitudes, weights=self.weights)
