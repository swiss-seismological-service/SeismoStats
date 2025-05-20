import warnings
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self

from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               shi_bolt_confidence)
from seismostats.utils._config import get_option
from seismostats.utils.binning import binning_test
from seismostats.analysis.bvalue.utils import ks_test_gr, ks_test_gr_lilliefors
from seismostats.utils.simulate_distributions import dither_magnitudes


class BValueEstimator(ABC):

    def __init__(self) -> Self:
        self._magnitudes: np.ndarray | None = None
        self._mc: float | None = None
        self._delta_m: float | None = None
        self._weights: np.ndarray | None = None
        self.idx: np.ndarray | None = None
        self.ks_ds: np.ndarray | None = None

        self.__b_value: float | None = None

    def calculate(self,
                  magnitudes: np.ndarray | list,
                  mc: float,
                  delta_m: float,
                  weights: np.ndarray | list | None = None) -> float:
        '''
        Calculates the b-value of the Gutenberg-Richter (GR) law.

        Args:
            magnitudes:     Array of magnitudes.
            mc:             Completeness magnitude.
            delta_m:        Bin size of discretized magnitudes.
            weights:        Array of weights for the magnitudes.

        Returns:
            b: b-value of the Gutenberg-Richter law.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> from seismostats.analysis import ClassicBValueEstimator

                >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6,
                ...                        2.3, 2.7, 2.2, 2.4, 2. , 2.7, 2.2,
                ...                        2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7,
                ...                        2.4, 2.2, 2.5])

                >>> my_estimator = ClassicBValueEstimator()
                >>> b_value = calculate(
                ...     magnitudes=magnitudes, mc=2.0, delta_m=0.1)

                >>> b_value

                1.114920128810535 # depending on the method used
        '''

        self.magnitudes = np.array(magnitudes)
        self.mc = mc
        self.delta_m = delta_m
        self.weights = None if weights is None else np.array(weights)

        self._sanity_checks()
        self._filter_magnitudes()

        if len(self.magnitudes) == 0:
            self.__b_value = np.nan
            return self.__b_value

        self.__b_value = self._estimate()
        return self.__b_value

    @abstractmethod
    def _estimate(self) -> float:
        '''
        Specific implementation of the b-value estimator.
        '''
        pass

    def _filter_magnitudes(self):
        '''
        Filters out magnitudes below the completeness magnitude.
        '''
        self.idx = (self.magnitudes >= self.mc - self.delta_m / 2).nonzero()[0]
        self.magnitudes = self.magnitudes[self.idx]

        if self.weights is not None:
            self.weights = self.weights[self.idx]

        if len(self.magnitudes) == 0:
            if get_option('warnings') is True:
                warnings.warn('No magnitudes above the completeness magnitude.')

    def _sanity_checks(self):
        '''
        Performs sanity checks on the input data.
        '''
        # test magnitude binnning
        if len(self.magnitudes) > 0:
            if not binning_test(self.magnitudes, self.delta_m,
                                check_larger_binning=False):
                raise ValueError('Magnitudes are not binned correctly.')

            # give warnings
            if get_option('warnings') is True:
                if np.min(self.magnitudes) - self.mc > self.delta_m / 2:
                    warnings.warn(
                        'No magnitudes in the lowest magnitude bin are present.'
                        'Check if mc is chosen correctly.'
                    )

        # test weights
        if self.weights is not None:
            if len(self.magnitudes) != len(self.weights):
                raise IndexError(
                    'The number of magnitudes and weights must be equal.')
            if np.any(self.weights < 0):
                raise ValueError('Weights must be nonnegative.')

    @classmethod
    @abstractmethod
    def _weights_supported(self) -> bool:
        '''
        Set to True if the estimator supports weights, False otherwise.
        '''
        pass

    @property
    def b_value(self) -> float:
        '''
        The b-value of the Gutenberg-Richter law.
        '''
        self.__is_estimated()
        return self.__b_value

    @property
    def beta(self) -> float:
        '''
        The beta value of the Gutenberg-Richter law.
        '''
        self.__is_estimated()
        return b_value_to_beta(self.__b_value)

    @property
    def std(self):
        '''
        Shi and Bolt uncertainty of the b-value estimate.
        '''
        self.__is_estimated()

        return shi_bolt_confidence(self.magnitudes,
                                   self.__b_value,
                                   weights=self.weights,
                                   b_parameter='b_value')

    @property
    def std_beta(self):
        '''
        Shi and Bolt uncertainty of the beta estimate.
        '''
        self.__is_estimated()

        return shi_bolt_confidence(self.magnitudes,
                                   self.beta,
                                   weights=self.weights,
                                   b_parameter='beta')

    @property
    def n(self):
        '''
        Number of magnitudes used to estimate the b-value.
        '''
        self.__is_estimated()
        return len(self.magnitudes)

    @property
    def p_ks(self):
        '''
        p-value of the ks-test.
        '''
        self.__is_estimated()
        out = ks_test_gr(self.magnitudes,
                         self.mc,
                         self.delta_m,
                         b_value=self.b_value,
                         ks_ds=self.ks_ds,
                         weights=self.weights)
        self.ks_ds = out[2]
        return out[0]

    @property
    def p_lilliefors(self):
        '''
        p-value of the Lilliefors test. Weights are not yet implemented.
        Magnitudes are dithered to continuous ones by resampling the
        distribution of the binned magnitudes, taking into account the
        exponential distribution of the magnitudes (in contrast to
        Herrmann et al).

        Source:
        - Herrmann, M. and W. Marzocchi (2020). "Inconsistencies and Lurking
        Pitfalls in the Magnitude-Frequency Distribution of High-Resolution
        Earthquake Catalogs".
        Seismological Research Letters 92(2A). doi: 10.1785/0220200337
        - Lilliefors, Hubert W. "On the Kolmogorov-Smirnov test for the
        exponential distribution with mean unknown." Journal of the American
        Statistical Association 64.325 (1969): 387-389.
        '''
        self.__is_estimated()

        # dither magnitudes to continuous ones
        n = 100
        p_vals = np.zeros(n)
        for ii in range(n):
            dithered_mags = dither_magnitudes(
                self.magnitudes, self.delta_m, self.b_value)
            p_vals[ii] = ks_test_gr_lilliefors(dithered_mags,
                                               self.mc - self.delta_m / 2)
        return np.mean(p_vals)

    def __is_estimated(self) -> bool:
        '''
        Checks if the b-value has been estimated.
        '''
        if self.__b_value is None:
            raise AttributeError('Please calculate the b value first.')

    @property
    def magnitudes(self) -> np.ndarray:
        '''
        The magnitudes used to estimate the b-value.
        '''
        return self._magnitudes

    @magnitudes.setter
    def magnitudes(self, magnitudes: np.ndarray) -> None:
        '''
        Sets the magnitudes used to estimate the b-value.
        '''
        self._magnitudes = magnitudes

    @property
    def mc(self) -> float:
        '''
        The completeness magnitude used to estimate the b-value.
        '''
        return self._mc

    @mc.setter
    def mc(self, mc: float) -> None:
        '''
        Sets the completeness magnitude.
        '''
        self._mc = mc

    @property
    def delta_m(self) -> float:
        '''
        Bin size of the discretized magnitudes.
        '''
        return self._delta_m

    @delta_m.setter
    def delta_m(self, delta_m: float) -> None:
        '''
        Sets the bin size of the discretized magnitudes.
        '''
        self._delta_m = delta_m

    @property
    def weights(self) -> np.ndarray | None:
        '''
        The weights used to estimate the b-value.
        '''
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray | None) -> None:
        '''
        Sets the weights used to estimate the b-value.
        '''

        self._weights = weights
