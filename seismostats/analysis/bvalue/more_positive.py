import warnings

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.utils._config import get_option
from seismostats.analysis.bvalue.utils import find_next_larger


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

        idx_next_larger = find_next_larger(
            self.magnitudes, self.delta_m, self.dmc)
        mag_diffs = self.magnitudes[idx_next_larger] - self.magnitudes

        idx_no_next = idx_next_larger == 0
        self.magnitudes = mag_diffs[~idx_no_next]

        if self.weights is not None:
            weights = self.weights[idx_next_larger]
            self.weights = weights[~idx_no_next]

        classic_estimator = ClassicBValueEstimator(mc=self.dmc,
                                                   delta_m=self.delta_m)

        return classic_estimator(self.magnitudes, weights=self.weights)
