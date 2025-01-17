import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from typing_extensions import Self

from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               shi_bolt_confidence)
from seismostats.utils._config import get_option
from seismostats.utils.binning import binning_test


class BValueEstimator(ABC):

    def __init__(self,
                 magnitudes: np.ndarray | list,
                 mc: float,
                 delta_m: float,
                 weights: np.ndarray | list | None = None) -> Self:

        # attributes with a getter and setter
        self.__magnitudes: np.ndarray | list | None = np.array(magnitudes)
        self.__weights: np.ndarray | list | None = \
            None if weights is None else np.array(weights)

        self.mc: float
        self._register_attribute('mc', mc)

        self.delta_m: float
        self._register_attribute('delta_m', delta_m)

        # internal attributes
        self.__b_value: float | None = None
        self.__std: float | None = None
        self.__beta: float | None = None
        self.__std_beta: float | None = None
        self.__used_magnitudes: np.ndarray = self.__magnitudes
        self.__used_weights: np.ndarray = self.__weights

    @abstractmethod
    def _estimate(self,
                  magnitudes: np.ndarray,
                  weights: np.ndarray | None
                  ) -> tuple[float, np.ndarray, np.ndarray | None]:
        '''
        Specific implementation of the b-value estimator.

        Args:
            magnitudes:         array of magnitudes
            weights:            array of weights

        Returns:
            b_value:            b-value estimate
            used_magnitudes:    magnitudes used for the estimate
            used_weights:       weights used for the estimate
        '''
        pass

    @classmethod
    @abstractmethod
    def weights_supported(self) -> bool:
        '''
        Set to True if the estimator supports weights, False otherwise.
        '''
        pass

    def __reset_estimates(self) -> None:
        '''
        Reset the estimates.
        '''
        self.__b_value = None
        self.__beta = None
        self.__std = None
        self.__std_beta = None
        self.__used_magnitudes = self.__magnitudes
        self.__used_weights = self.__weights

    def b_value(self) -> float:
        '''
        Estimate the b-value of the Gutenberg-Richter law.
        '''
        if self.__b_value is not None:
            return self.__b_value

        self._sanity_checks()

        self.__b_value, self.__used_magnitudes, self.__used_weights = \
            self._estimate(
                self.__magnitudes.copy(),
                self.__weights.copy() if self.__weights is not None else None)

        return self.__b_value

    def beta(self) -> float:
        '''
        Estimate the beta value of the Gutenberg-Richter law.
        '''
        if self.__beta is None:
            self.__beta = b_value_to_beta(self.b_value())
        return self.__beta

    def std(self):
        '''
        Shi and Bolt estimate of the b-value estimate.
        '''
        if self.__std is None:
            self.__std = self.__calculate_std('b_value')
        return self.__std

    def std_beta(self):
        '''
        Shi and Bolt estimate of the beta estimate.
        '''
        if self.__std_beta is None:
            self.__std_beta = self.__calculate_std('beta')
        return self.__std_beta

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
            binning_test(self.__magnitudes, self.delta_m, tolerance)
        )
        'Magnitudes are not binned correctly.'

        if self.__weights is not None:
            assert len(self.__magnitudes) == len(self.__weights), (
                'The number of magnitudes and weights must be equal.'
            )
            assert np.all(self.__weights >= 0), 'Weights must be nonnegative.'

        # test if lowest magnitude is much larger than mc
        if get_option('warnings') is True:
            if np.min(self.__magnitudes) - self.mc > self.delta_m / 2:
                warnings.warn(
                    'No magnitudes in the lowest magnitude bin are present. '
                    'Check if mc is chosen correctly.'
                )

    @property
    def magnitudes(self):
        return self.__used_magnitudes

    @magnitudes.setter
    def magnitudes(self, magnitudes):
        self.__magnitudes = magnitudes
        self.__reset_estimates()

    @property
    def weights(self):
        return self.__used_weights

    @weights.setter
    def weights(self, weights):
        if not self.weights_supported and weights is not None:
            raise ValueError('Weights are not supported by this estimator.')
        self.__weights = weights
        self.__reset_estimates()

    @property
    def n(self):
        if self.__b_value is None:
            raise AttributeError('Please run the estimator first.')
        return len(self.__used_magnitudes)

    def _register_attribute(self, name: str, value: Any):
        '''
        Register an attribute to the estimator.
        '''
        private_attr = f'__{name}'

        setattr(self, private_attr, value)

        def getter(self):
            return getattr(self, private_attr)

        def setter(self, value):
            setattr(self, private_attr, value)
            self.__reset_estimates()

        # Create the property
        setattr(self.__class__, name, property(getter, setter))

    def __calculate_std(self, parameter: Literal['b_value', 'beta']):
        '''
        Calculate the Shi and Bolt estimate of the standard deviation of the
        b-value/beta estimate.
        '''
        if self.__b_value is None:
            raise AttributeError('Please run the estimator first.')

        if get_option('warnings') is True:
            if self.weights is not None:
                warnings.warn(
                    'Shi and Bolt confidence with weights considers the '
                    'magnitudes as '
                    'having length {}, the sum of relevant weights.'.format(
                        np.sum(self.__used_weights))
                )

        b = self.b_value() if parameter == 'b_value' else self.beta()
        return shi_bolt_confidence(self.__used_magnitudes,
                                   weights=self.__used_weights,
                                   b=b,
                                   b_parameter=parameter)
