import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from typing_extensions import Self

from seismostats.analysis.bvalue.utils import (b_value_to_beta,
                                               shi_bolt_confidence)
from seismostats.utils._config import get_option
from seismostats.utils.binning import binning_test


class BValueEstimator(ABC):

    def __init__(self,
                 mc: float,
                 delta_m: float) -> Self:
        self.mc = mc
        self.delta_m = delta_m

        self.__b_value = None
        self.__b_parameter: Literal['b_value', 'beta'] = 'b_value'

    @abstractmethod
    def _estimate(self):
        """
        Specific implementation of the b-value estimator.
        """
        pass

    @classmethod
    @abstractmethod
    def weights_supported(self):
        """
        Set to True if the estimator supports weights, False otherwise.
        """
        pass

    def __call__(self,
                 magnitudes: np.ndarray | list,
                 weights: np.ndarray | list | None = None) -> float:

        if not self.weights_supported and weights is not None:
            raise ValueError('Weights are not supported by this estimator.')

        self.magnitudes = magnitudes.copy()

        self.weights = weights

        self._sanity_checks()

        self.__b_value = self._estimate()

        return self.__b_value

    def estimate_beta(self,
                      magnitudes: np.ndarray | list,
                      weights: np.ndarray | list | None = None) -> float:
        '''
        Estimate the beta value of the Gutenberg-Richter law using the
        maximum likelihood estimator.
        '''
        self.__b_value = self.__call__(magnitudes, weights)
        self.__b_parameter = 'beta'
        return b_value_to_beta(self.__b_value)

    @property
    def std(self):
        '''
        Shi and Bolt estimate of the beta/b-value estimate.
        '''
        assert self.__b_value is not None, 'Please run the estimator first.'
        if get_option('warnings') is True:
            if self.weights is not None:
                warnings.warn(
                    'Shi and Bolt confidence with weights considers the '
                    'magnitudes as '
                    'having length {}, the sum of relevant weights.'.format(
                        np.sum(self.weights))
                )
        return shi_bolt_confidence(self.magnitudes,
                                   weights=self.weights,
                                   b=self.__b_value,
                                   b_parameter=self.__b_parameter)

    @property
    def n(self):
        return len(self.magnitudes)

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
        "Magnitudes are not binned correctly."

        if self.weights is not None:
            assert len(self.magnitudes) == len(self.weights), (
                "The number of magnitudes and weights must be equal."
            )
            assert np.all(self.weights >= 0), "Weights must be nonnegative."

        # test if lowest magnitude is much larger than mc
        if get_option("warnings") is True:
            if np.min(self.magnitudes) - self.mc > self.delta_m / 2:
                warnings.warn(
                    "No magnitudes in the lowest magnitude bin are present. "
                    "Check if mc is chosen correctly."
                )
