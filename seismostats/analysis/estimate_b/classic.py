import numpy as np

from seismostats.analysis.estimate_b.base import BValueEstimator
from seismostats.analysis.estimate_b.utils import beta_to_b_value


class ClassicBValueEstimator(BValueEstimator):

    weights_supported = True

    def __init__(self, *args, **kwargs):
        '''Return the maximum likelihood b-value or beta for
        an array of magnitudes and a completeness magnitude mc.
        If the magnitudes are discretized, the discretization must be given in
        ``delta_m``, so that the maximum likelihood estimator can be calculated
        correctly.


        Source:
            - Aki 1965 (Bull. Earthquake research institute, vol 43, pp 237-239)
            - Tinti and Mulargia 1987 (Bulletin of the Seismological Society of
            America, 77(6), 2125-2134.)

        Args:
            mc:         completeness magnitude
            delta_m:    Discretization of magnitudes.
                        Default is no discretization.
        '''
        super().__init__(*args, **kwargs)

    def _estimate(self):
        if self.delta_m > 0:
            p = 1 + self.delta_m / \
                np.average(self.magnitudes - self.mc, weights=self.weights)
            beta = 1 / self.delta_m * np.log(p)
        else:
            beta = 1 / np.average(self.magnitudes
                                  - self.mc, weights=self.weights)

        return beta_to_b_value(beta)


class UtsuBValueEstimator(BValueEstimator):

    weights_supported = False

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
        beta = 1 / np.mean(self.magnitudes - self.mc + self.delta_m / 2)
        return beta_to_b_value(beta)
