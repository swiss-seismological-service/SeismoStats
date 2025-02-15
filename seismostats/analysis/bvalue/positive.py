import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.utils._config import get_option


class BPositiveBValueEstimator(BValueEstimator):
    '''
    B-value estimator using positive differences between
    consecutive magnitudes.

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126,
        Issue 2)
    '''

    weights_supported = True

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  times: np.ndarray | None = None,
                  weights: np.ndarray | None = None,
                  dmc: float | None = None) -> float:
        '''
        Returns the b-value estimate calculated using the
        positive differences between consecutive magnitudes.

        Args:
            magnitudes: Array of magnitudes
            mc:         Completeness magnitude
            delta_m:    Discretization of magnitudes.
            times:      Vector of times of the events, in any format (datetime,
                    float, etc.). If None, it is assumed that the events are
                    ordered in time.
            weights:    Array of weights for the magnitudes.
            dmc:        Cutoff value for the differences (differences below
                    this value are not considered). If None, the cutoff is set
                    to delta_m.
        '''
        self.times: np.ndarray | None = np.array(
            times) if times is not None else times
        self.dmc: float = dmc if dmc is not None else delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0')

        if self.dmc < delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended')

        return super().calculate(magnitudes,
                                 mc=mc,
                                 delta_m=delta_m,
                                 weights=weights)

    def _filter_magnitudes(self) -> np.ndarray:
        '''
        Filter out magnitudes below the completeness magnitude.
        '''
        super()._filter_magnitudes()
        if self.times is not None:
            self.times = self.times[self.idx]
        return self.idx

    def _estimate(self) -> float:
        # order the magnitudes and times
        if self.times is not None:
            srt = np.argsort(self.times)
            self.magnitudes = self.magnitudes[srt]
            self.times = self.times[srt]
            if self.weights is not None:
                self.weights = self.weights[srt]
            self.idx = self.idx[srt]

        # calculate differences, only keep positive ones
        self.magnitudes = np.diff(self.magnitudes)
        is_larger = self.magnitudes >= self.dmc - self.delta_m / 2
        self.magnitudes = self.magnitudes[is_larger]
        self.idx = self.idx[1:][is_larger]

        if self.weights is not None:
            # use weight of second earthquake of a difference
            self.weights = self.weights[1:][is_larger]
        if self.times is not None:
            self.times = self.times[1:][is_larger]

        return _mle_estimator(self.magnitudes,
                              mc=self.dmc,
                              delta_m=self.delta_m,
                              weights=self.weights)
