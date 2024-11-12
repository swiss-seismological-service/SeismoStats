from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from seismostats.analysis.b_values.utils import b_value_to_beta
from seismostats.analysis.estimate_beta import shi_bolt_confidence


class BValueEstimator(ABC):

    def __init__(self,
                 mc: float,
                 delta_m: float):
        self.mc = mc
        self.delta_m = delta_m

        self.__b_value = None
        self.__b_parameter: Literal['b_value', 'beta'] = 'b_value'

    @abstractmethod
    def _estimate(self):
        pass

    @classmethod
    @abstractmethod
    def weights_supported(self):
        pass

    def __call__(self,
                 magnitudes: np.ndarray | list,
                 weights: np.ndarray | list | None = None) -> float:

        if not self.weights_supported and weights is not None:
            raise ValueError('Weights are not supported by this estimator')

        self.magnitudes = magnitudes
        self.weights = weights

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
        assert self.__b_value is not None, 'Please run the estimator first'
        return shi_bolt_confidence(self.magnitudes,
                                   b=self.__b_value,
                                   b_parameter=self.__b_parameter)

    @property
    def n(self):
        return len(self.magnitudes)
