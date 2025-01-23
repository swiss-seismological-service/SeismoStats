import warnings
from seismostats.analysis.avalue.base import AValueEstimator
import numpy as np

from seismostats.utils._config import get_option


class ClassicAValueEstimator(AValueEstimator):
    '''
    Estimator for the a-value of the Gutenberg-Richter (GR) law.
    '''

    weights_supported = False

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        # TODO should self.delta_m and self.magnitudes be updated?
        if self.delta_m is None:
            self.delta_m = 0
        if self.mc is None:
            self.mc = self.magnitudes.min()
        elif self.magnitudes.min() < self.mc - self.delta_m / 2:
            if get_option("warnings") is True:
                warnings.warn(
                    "Completeness magnitude is higher than"
                    "the lowest magnitude."
                    "Cutting the magnitudes to the completeness magnitude.")
            self.magnitudes = self.magnitudes[self.magnitudes
                                              >= self.mc - self.delta_m / 2]

        a = np.log10(len(self.magnitudes))
        a = self._reference_scaling(a)
        return a
