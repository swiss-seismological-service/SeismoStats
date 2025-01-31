import warnings
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self

from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               shi_bolt_confidence)
from seismostats.utils._config import get_option
from seismostats.utils.binning import binning_test


class BValueEstimator(ABC):

    def __init__(self) -> Self:
        self.magnitudes: np.ndarray | None = None
        self.mc: float | None = None
        self.delta_m: float | None = None
        self.weights: np.ndarray | None = None
        self.idx: np.ndarray | None = None

        self.__b_value: float | None = None

    def calculate(self,
                  magnitudes: np.ndarray | list,
                  mc: float,
                  delta_m: float,
                  weights: np.ndarray | list | None = None) -> float:
        '''
        Return the b-value estimate.

        Args:
            magnitudes: Array of magnitudes
            mc:         Completeness magnitude
            delta_m:    Discretization of magnitudes.
            weights:    Array of weights for the magnitudes.

        Returns:
            b: b-value of the Gutenberg-Richter law.
        '''

        self.magnitudes = np.array(magnitudes)
        self.mc = mc
        self.delta_m = delta_m
        self.weights = None if weights is None else np.array(weights)

        self._filter_magnitudes()
        self._sanity_checks()

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
        Filter out magnitudes below the completeness magnitude.
        '''
        idx = self.magnitudes >= self.mc - self.delta_m / 2
        self.magnitudes = self.magnitudes[idx]

        if self.weights is not None:
            self.weights = self.weights[idx]

        return idx

    def _sanity_checks(self):
        '''
        Perform sanity checks on the input data.
        '''

        # test that the magnitudes are binned correctly
        if self.delta_m == 0:
            tolerance = 1e-08
        else:
            tolerance = max(self.delta_m / 100, 1e-08)
        assert (
            binning_test(self.magnitudes, self.delta_m, tolerance)
        )
        'Magnitudes are not binned correctly.'

        if self.weights is not None:
            assert len(self.magnitudes) == len(self.weights), (
                'The number of magnitudes and weights must be equal.'
            )
            assert np.all(self.weights >= 0), 'Weights must be nonnegative.'

        # test if lowest magnitude is much larger than mc
        if get_option('warnings') is True:
            if np.min(self.magnitudes) - self.mc > self.delta_m / 2:
                warnings.warn(
                    'No magnitudes in the lowest magnitude bin are present. '
                    'Check if mc is chosen correctly.'
                )

    @classmethod
    @abstractmethod
    def weights_supported(self) -> bool:
        '''
        Set to True if the estimator supports weights, False otherwise.
        '''
        pass

    @property
    def b_value(self) -> float:
        '''
        Returns the b value of the Gutenberg-Richter law.
        '''
        self.__is_estimated()
        return self.__b_value

    @property
    def beta(self) -> float:
        '''
        Returns the beta value of the Gutenberg-Richter law.
        '''
        self.__is_estimated()
        return b_value_to_beta(self.__b_value)

    @property
    def std(self):
        '''
        Shi and Bolt estimate of the b-value estimate.
        '''
        self.__is_estimated()

        return shi_bolt_confidence(self.magnitudes,
                                   self.__b_value,
                                   weights=self.weights,
                                   b_parameter='b_value')

    @property
    def std_beta(self):
        '''
        Shi and Bolt estimate of the beta estimate.
        '''
        self.__is_estimated()

        return shi_bolt_confidence(self.magnitudes,
                                   self.beta,
                                   weights=self.weights,
                                   b_parameter='beta')

    @property
    def n(self):
        self.__is_estimated()
        return len(self.magnitudes)

    def __is_estimated(self) -> bool:
        '''
        Check if the b-value has been estimated.
        '''
        if self.__b_value is None:
            raise AttributeError('Please calculate the b value first.')
