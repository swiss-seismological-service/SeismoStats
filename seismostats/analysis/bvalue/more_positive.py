import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.analysis.bvalue.utils import find_next_larger
from seismostats.utils._config import get_option


class BMorePositiveBValueEstimator(BValueEstimator):
    """
    B-value estimator using the next positive differences (this means that
    almost every magnitude has a difference, as opposed to the b-positive
    method which results in half the data).

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    """

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
        Return the b-value estimate calculated using the
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
        if self.times is not None:
            srt = np.argsort(self.times)
            self.magnitudes = self.magnitudes[srt]
            self.times = self.times[srt]
            if self.weights is not None:
                self.weights = self.weights[srt]
            self.idx = self.idx[srt]

        # calculate mg diffs to next larger magnitude
        mag_diffs = -np.ones(len(self.magnitudes) - 1) * self.delta_m
        idx_next_larger = find_next_larger(
            self.magnitudes, self.delta_m, self.dmc)
        mag_diffs = self.magnitudes[idx_next_larger] - self.magnitudes

        idx_no_next = idx_next_larger == 0
        self.magnitudes = mag_diffs[~idx_no_next]

        # make sure that all attributes are consistent
        idx_next_larger = idx_next_larger[~idx_no_next]
        self.idx = self.idx[idx_next_larger]
        if self.weights is not None:
            self.weights = self.weights[idx_next_larger]
        if self.times is not None:
            self.times = self.times[idx_next_larger]

        return _mle_estimator(self.magnitudes,
                              mc=self.dmc,
                              delta_m=self.delta_m,
                              weights=self.weights)
