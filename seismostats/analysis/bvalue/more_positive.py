import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.analysis.bvalue.utils import find_next_larger
from seismostats.utils._config import get_option


class BMorePositiveBValueEstimator(BValueEstimator):
    """
    Returns the b-value of the Gutenberg-Richter (GR) law using earthquake
    pairs for which the latter is larger than the former
    by some margin, :math:`m_j \ge m_{i} + dmc`.

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis.bvalue import BMorePositiveBValueEstimator

            >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3, 2.7, 2.2, 2.4, 2. , 2.7, 2.2, 2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5])
            >>> mc = 2.0
            >>> delta_m = 0.1
            >>> dmc = 0.2

            >>> my_estimator = BMorePositiveBValueEstimator()
            >>> b_value = my_estimator.calculate(magnitudes=magnitudes, mc=mc, delta_m=delta_m, dmc=dmc)

            >>> print(b_value)

            2.253092817258629

        .. code-block:: python

            .. code-block:: python

            >>> print("used magnitudes:      ", my_estimator.magnitudes)
            >>> print("used mc:              ", my_estimator.mc)
            >>> print("used delta_m:         ", my_estimator.delta_m)
            >>> print("used dmc:             ", my_estimator.dmc)
            >>> print("b-value:              ", my_estimator.b_value)
            >>> print("b-value uncertainty:  ", my_estimator.std)

            used magnitudes:       [0.5 0.2 0.4 0.3 0.2 0.4 0.4 0.2 0.3 0.7 0.2 0.3 0.3 0.2 0.5 0.5 0.3]
            used mc:               2.0
            used delta_m:          0.1
            used dmc:              0.2
            b-value:               2.253092817258629
            b-value uncertainty:   0.402397451336126

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
        Args:
            magnitudes: Array of magnitudes
            mc:         Completeness magnitude.
            delta_m:    Bin size of discretized magnitudes.
            times:      Array of times of the events, in any format (datetime,
                    float, etc.). If `None`, it is assumed that the events are
                    ordered in time.
            weights:    Array of weights for the magnitudes.
            dmc:        Cutoff value for the differences (differences below
                    this value are not considered). If `None`, the cutoff is set
                    to `delta_m`.
        Returns:
            b: b-value of the Gutenberg-Richter law.
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
        Filters out magnitudes below the completeness magnitude.
        '''
        super()._filter_magnitudes()
        if self.times is not None:
            self.times = self.times[self.idx]

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
