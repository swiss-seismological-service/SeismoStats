# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
# Hazard Modeller's Toolkit (openquake.hmtk) https://www.globalquakemodel.org

import abc

import numpy as np
import pandas as pd

DistanceTimeWindow = tuple[np.ndarray[float], np.ndarray[pd.Timedelta]]
_DistanceTimeWindow = tuple[np.ndarray[float], np.ndarray[float]]


class BaseDistanceTimeWindow(abc.ABC):
    """
    Defines the space and time windows, within which an event is identified
    as a cluster.
    """

    def __init__(self, time_cutoff: float | None = None):
        """
        Args:
            time_cutoff:    Time window cutoff in days (optional).
                        No time windows larger than time_cutoff are returned.
        """
        self.time_cutoff = time_cutoff

    @abc.abstractmethod
    def _calc(
        self,
        magnitude: np.ndarray,
    ) -> _DistanceTimeWindow:
        """
        Calculates the space and time windows for given magnitudes.

        Args:
            magnitude:  Array of magnitudes.

        Returns:
            sw_space:   Array of space windows in km.
            sw_time:    Array of time windows as decimal days.
        """
        return NotImplemented

    def __call__(self, magnitude: np.ndarray):
        """
        Calculate the space and time windows for given magnitudes with cutoff.

        Args:
            magnitude:  Array of magnitudes.

        Returns:
            sw_space:   Array of space windows in km.
            sw_time:    Array of time windows.
        """
        sw_space, sw_time = self._calc(magnitude)
        if self.time_cutoff:
            sw_time = np.clip(sw_time, a_min=0, a_max=self.time_cutoff)
        sw_time = np.array([pd.Timedelta(days=t) for t in sw_time])
        return sw_space, sw_time


class GardnerKnopoffWindow(BaseDistanceTimeWindow):
    """
    Gardner Knopoff method for calculating distance and time windows.

    Source:
        Gardner, J. K. and Knopoff, L. (1974). Is the sequence of aftershocks
        in Southern California, with aftershocks removed, poissonian?. Bull.
        Seism. Soc. Am., 64(5): 1363-1367.
    """

    def _calc(self, magnitude: np.ndarray) -> _DistanceTimeWindow:
        sw_space = np.power(10.0, 0.1238 * magnitude + 0.983)
        sw_time = np.power(10.0, 0.032 * magnitude + 2.7389)
        sw_time[magnitude < 6.5] = np.power(
            10.0, 0.5409 * magnitude[magnitude < 6.5] - 0.547
        )
        return sw_space, sw_time


class GruenthalWindow(BaseDistanceTimeWindow):
    """
    Gruenthal method for calculating distance and time windows.

    Source:
        Gruenthal, G. (1985) The up-dated earthquake catalogue for the German
        Democratic Republic and adjacent areas - statistical data
        characteristics and conclusions for hazard assessment. 3rd
        International Symposium on the Analysis of Seismicity and Seismic
        Risk, Liblice/Czechoslovakia, 17–22 June 1985 (Proceedings Vol. I,
        19–25)

        See also:
        van Stiphout, T., Zhuang, J., & Marsan, D. (2012). Seismicity
        declustering. Community online resource for statistical seismicity
        analysis, 10(1), 1-25.


    """

    def _calc(self, magnitude: np.ndarray) -> _DistanceTimeWindow:
        sw_space = np.exp(1.77 + np.sqrt(0.037 + 1.02 * magnitude))
        sw_time = np.abs((np.exp(-3.95 + np.sqrt(0.62 + 17.32 * magnitude))))
        sw_time[magnitude >= 6.5] = np.power(
            10, 2.8 + 0.024 * magnitude[magnitude >= 6.5]
        )
        return sw_space, sw_time


class UhrhammerWindow(BaseDistanceTimeWindow):
    """
    Uhrhammer method for calculating distance and time windows.

    Source:
        Uhrhammer, R. Characteristics of Northern and Central California
        seismicity, Earthquake Notes, 1986, vol. 57, no. 1, p. 21.
    """

    def _calc(self, magnitude: np.ndarray) -> _DistanceTimeWindow:
        sw_space = np.exp(-1.024 + 0.804 * magnitude)
        sw_time = np.exp(-2.87 + 1.235 * magnitude)
        return sw_space, sw_time
