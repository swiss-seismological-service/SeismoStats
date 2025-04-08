import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision


class BPositiveBValueEstimator(BValueEstimator):
    '''
    Estimator to calculate the b-value and other parameters using only the
    earthquakes with magnitudes :math:`m_i \\ge m_{i-1} + dmc`.

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126,
        Issue 2)

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis import BPositiveBValueEstimator

            >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3,
            ...                        2.7, 2.2, 2.4, 2. , 2.7, 2.2, 2.3, 2.1,
            ...                        2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5])

            >>> my_estimator = BPositiveBValueEstimator()
            >>> my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1, dmc=0.2)

            >>> my_estimator.b_value

            1.9188552623891313
    '''

    _weights_supported = True

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  weights: np.ndarray | None = None,
                  times: np.ndarray | None = None,
                  dmc: float | None = None) -> float:
        '''
        Calculates the b-value of the Gutenberg-Richter (GR) law.

        Args:
            magnitudes: Array of magnitudes.
            mc:         Completeness magnitude.
            delta_m:    Bin size of discretized magnitudes.
            weights:    Array of weights for the magnitudes.
            times:      Array of times of the events, in any format (datetime,
                    float, etc.). If `None`, it is assumed that the events are
                    ordered in time.
            dmc:        Cutoff value for the differences (differences below
                    this value are not considered). If `None`, the cutoff is set
                    to `delta_m`.

        Returns:
            b:          b-value of the Gutenberg-Richter law.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> from seismostats.analysis import BPositiveBValueEstimator

                >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6,
                ...                        2.3, 2.7, 2.2, 2.4, 2. , 2.7, 2.2,
                ...                        2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7,
                ...                        2.4, 2.2, 2.5])

                >>> my_estimator = BPositiveBValueEstimator()
                >>> b_value = my_estimator.calculate(
                ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1, dmc=0.2)

                >>> b_value

                1.9188552623891313
        '''
        self.times: np.ndarray | None = np.array(
            times) if times is not None else times
        self.dmc: float = dmc if dmc is not None else delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0.')

        if self.dmc < delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended.')

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
        # order the magnitudes and times
        if self.times is not None:
            srt = np.argsort(self.times)
            self.magnitudes = self.magnitudes[srt]
            self.times = self.times[srt]
            if self.weights is not None:
                self.weights = self.weights[srt]
            self.idx = self.idx[srt]

        # calculate differences, only keep positive ones
        mag_diffs = np.diff(self.magnitudes)
        is_larger = mag_diffs >= self.dmc - self.delta_m / 2
        mag_diffs = bin_to_precision(mag_diffs, self.delta_m)
        self.magnitudes = mag_diffs[is_larger]
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

    @property
    def dmc(self) -> float:
        '''
        The dmc value used for the calculation.
        '''
        return self._dmc

    @dmc.setter
    def dmc(self, dmc: float) -> None:
        '''
        Sets the dmc value used for the calculation.
        '''
        self._dmc = dmc

    @property
    def magnitudes(self) -> np.ndarray:
        '''
        The positive magnitude differences used for the calculation.
        '''
        return self._magnitudes

    @magnitudes.setter
    def magnitudes(self, magnitudes: np.ndarray) -> None:
        '''
        Sets the positive magnitude differences used for the calculation.
        '''
        self._magnitudes = magnitudes
