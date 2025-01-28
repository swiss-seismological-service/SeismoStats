from seismostats.analysis.avalue.base import AValueEstimator
import numpy as np


class ClassicAValueEstimator(AValueEstimator):
    '''
    Estimator for the a-value of the Gutenberg-Richter (GR) law.
    '''

    weights_supported = False

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        a = np.log10(len(self.magnitudes))
        a = self._reference_scaling(a)
        return a
