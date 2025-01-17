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

        mag_diffs = -np.ones(len(magnitudes) - 1) * self.delta_m

        if weights is not None:
            weights_pos = np.ones_like(mag_diffs)

        for ii in range(len(magnitudes) - 1):
            for jj in range(ii + 1, len(magnitudes)):
                mag_diff_loop = magnitudes[jj] - magnitudes[ii]
                if mag_diff_loop > self.dmc - self.delta_m / 2:
                    mag_diffs[ii] = mag_diff_loop
                    if weights is not None:
                        # use weight of second earthquake of a difference
                        weights_pos[ii] = weights[jj]
                    break

        # only take the values where the next earthquake is larger
        is_larger = mag_diffs > -self.delta_m / 2

        if weights is not None:
            weights = weights_pos[is_larger]

        magnitudes = abs(mag_diffs[is_larger])

        classic_estimator = ClassicBValueEstimator(magnitudes,
                                                   mc=self.dmc,
                                                   delta_m=self.delta_m,
                                                   weights=weights)

        return classic_estimator.b_value(), magnitudes, weights
