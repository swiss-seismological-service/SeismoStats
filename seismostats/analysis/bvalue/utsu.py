import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.utils import beta_to_b_value


class UtsuBValueEstimator(BValueEstimator):

    weights_supported = True

    def __init__(self, *args, **kwargs):
        """Return the maximum likelihood b-value or beta.

        Source:
            Utsu 1965 (Geophysical bulletin of the Hokkaido University,
            vol 13, pp 99-103).

        Args:
            mc:         completeness magnitude
            delta_m:    discretization of magnitudes.
                        default is no discretization
        """

        super().__init__(*args, **kwargs)

    def _estimate(self):
        beta = 1 / np.average(self.magnitudes - self.mc
                              + self.delta_m / 2, weights=self.weights)
        return beta_to_b_value(beta)
