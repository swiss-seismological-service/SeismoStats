import numpy as np

from seismostats.analysis.avalue.base import AValueEstimator


class ClassicAValueEstimator(AValueEstimator):
    '''
    Return the a-value of the Gutenberg-Richter (GR) law.

    .. math::
        N(m) = 10 ^ {a - b \\cdot (m - m_{ref})}

    where N(m) is the number of events with magnitude larger or equal to m
    that occurred in the timeframe of the catalog.
    '''

    def __init__(self):
        super().__init__()

    def _estimate(self) -> float:
        return np.log10(len(self.magnitudes))
