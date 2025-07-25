import warnings

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import _mle_estimator
from seismostats.analysis.bvalue.utils import find_next_larger, b_value_to_beta
from seismostats.utils._config import get_option
from seismostats.utils.binning import bin_to_precision
from seismostats.analysis.bvalue.utils import bootstrap_std


class BMorePositiveBValueEstimator(BValueEstimator):
    '''
    Estimator to calculate the b-value and other parameters of the
    Gutenberg-Richter (GR) law using earthquake pairs for which
    the latter is the first one that is larger than the former by some margin,
    :math:`m_j \\ge m_{i} + dmc`.

    Source:
        E. Lippiello and G. Petrillo. Journal of Geophysical Research: Solid
        Earth, 129(2):e2023JB027849, 2024.

    Examples:
        .. code-block:: python

            >>> import numpy as np
            >>> from seismostats.analysis import BMorePositiveBValueEstimator

            >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3,
            ...                        2.7, 2.2, 2.4, 2. , 2.7, 2.2, 2.3, 2.1,
            ...                        2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5])

            >>> my_estimator = BMorePositiveBValueEstimator()
            >>> my_estimator.calculate(
            ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1, dmc=0.2)

            >>> my_estimator.b_value

            2.253092817258629
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
            dmc:        Margin by which the latter magnitude has to be
                    larger than the former. If `None`, the default value is
                    `delta_m`.

        Returns:
            b: b-value of the Gutenberg-Richter law.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> from seismostats.analysis.bvalue import \
                ...     BMorePositiveBValueEstimator

                >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6,
                ...                        2.3, 2.7, 2.2, 2.4, 2. , 2.7, 2.2,
                ...                        2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7,
                ...                        2.4, 2.2, 2.5])

                >>> my_estimator = BMorePositiveBValueEstimator()
                >>> b_value = my_estimator.calculate(
                ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1, dmc=0.2)

                >>> print(b_value)

                2.253092817258629
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
            self._original_mags = self._original_mags[srt]
            self.times = self.times[srt]
            if self.weights is not None:
                self.weights = self.weights[srt]
                self._original_weights = self._original_weights[srt]
            self.idx = self.idx[srt]

        # calculate mg diffs to next larger magnitude
        mag_diffs = -np.ones(len(self.magnitudes) - 1) * self.delta_m
        idx_next_larger = find_next_larger(
            self.magnitudes, self.delta_m, self.dmc)
        mag_diffs = self.magnitudes[idx_next_larger] - self.magnitudes
        mag_diffs = bin_to_precision(mag_diffs, self.delta_m)

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

    def _std_bootstrap(self, n: int = 500, random_state: int = None) -> float:
        '''
        Bootstrap uncertainty of the b-value estimate. Depending on the number
        of the magnitudes, this can take a while to compute.
        '''
        estimator = self.__class__()

        # Define the function for the bootstrap.
        def func(idx):
            idx = np.sort(idx)
            magnitudes = self._original_mags[idx]
            if self.weights is not None:
                weights = self._original_weights[idx]
            else:
                weights = None
            estimator.calculate(magnitudes, self.mc, self.delta_m,
                                weights=weights, dmc=self.dmc)
            return estimator.b_value

        # This takes bootstrap samples of the index and then calculates
        # the variance of the output of func (above)
        std = bootstrap_std(np.arange(len(self._original_mags)),
                            func, n=n, random_state=random_state)
        return std

    @property
    def std(self) -> float:
        '''
        Bootstrap uncertainty of the b-value estimate. Depending on the number
        of the magnitudes, this can take a while to compute.
        '''
        self._is_estimated()

        return self._std_bootstrap()

    @property
    def std_beta(self) -> float:
        '''
        Bootstrap uncertainty of the beta estimate. Depending on the number
        of the magnitudes, this can take a while to compute.
        '''
        self._is_estimated()

        std = self._std_bootstrap()
        return b_value_to_beta(std)

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
