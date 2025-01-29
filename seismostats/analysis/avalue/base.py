import warnings
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self

from seismostats.utils._config import get_option

# from seismostats.utils.binning import binning_test


class AValueEstimator(ABC):

    def __init__(self) -> Self:
        self.magnitudes: np.ndarray | None = None
        self.mc: float | None = None
        self.delta_m: float | None = None
        self.scaling_factor: float | None = None
        self.m_ref: float | None = None
        self.b_value: float | None = None

        self.__a_value: float | None = None

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  *args,
                  scaling_factor: float | None = None,
                  m_ref: float | None = None,
                  b_value: float | None = None,
                  **kwargs) -> float:
        '''
        Return the a-value estimate.

        Args:
            magnitudes:     Array of magnitudes
            mc:             Completeness magnitude
            delta_m:        Discretization of magnitudes.
            scaling_factor: Scaling factor
                            If given, this is used to normalize the number of
                            observed events. For example: Volume or area of the
                            region considered or length of the time interval,
                            given in the unit of interest.
            m_ref:          Reference magnitude for which the a-value
                            is estimated.
            b_value:        B-value of the Gutenberg-Richter law. Only relevant
                            when m_ref is not None.

        Returns:
            a: a-value of the Gutenberg-Richter law.
        '''

        self.magnitudes = np.array(magnitudes)
        self.mc = mc
        self.delta_m = delta_m
        self.scaling_factor = scaling_factor
        self.m_ref = m_ref
        self.b_value = b_value

        self._sanity_checks()
        self._filtering()

        self.__a_value = self._estimate(*args, **kwargs)
        self.__a_value = self._reference_scaling(self.__a_value)

        return self.__a_value

    @abstractmethod
    def _estimate(self, *args, **kwargs) -> float:
        '''
        Specific implementation of the a-value estimator.
        '''
        pass

    def _filtering(self) -> np.ndarray:
        '''
        Filter out magnitudes below the completeness magnitude.
        '''
        self.idx = self.magnitudes >= self.mc - self.delta_m / 2
        self.magnitudes = self.magnitudes[self.idx]

    def _sanity_checks(self):
        '''
        Perform sanity checks on the input data.
        '''
        # TODO test that the magnitudes are binned correctly
        # if self.delta_m == 0:
        #     tolerance = 1e-08
        # else:
        #     tolerance = max(self.delta_m / 100, 1e-08)
        # assert (
        #     binning_test(self.magnitudes, self.delta_m, tolerance)
        # )
        # 'Magnitudes are not binned correctly.'

        # test if lowest magnitude is much larger than mc
        if get_option('warnings') is True:
            if np.min(self.magnitudes) - self.mc > self.delta_m / 2:
                warnings.warn(
                    'No magnitudes in the lowest magnitude bin are present. '
                    'Check if mc is chosen correctly.'
                )

    def _reference_scaling(self, a: float) -> float:
        '''
        Args:
            a: a-value

        Returns:
            a: scaled a-value
        '''
        # scale to reference magnitude
        if self.m_ref is not None:
            if self.b_value is None:
                raise ValueError(
                    "b_value must be provided if m_ref is given")
            a = a - self.b_value * (self.m_ref - self.mc)

        # scale to reference time-interal or volume of interest
        if self.scaling_factor is not None:
            a = a - np.log10(self.scaling_factor)
        return a

    @property
    def a_value(self) -> float:
        '''
        Returns the a value of the Gutenberg-Richter law.
        '''
        if self.__a_value is None:
            raise AttributeError('Please calculate the a value first.')
        return self.__a_value
