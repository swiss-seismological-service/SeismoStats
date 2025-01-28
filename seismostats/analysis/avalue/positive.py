import warnings
from seismostats.analysis.avalue.base import AValueEstimator
import numpy as np

from seismostats.utils._config import get_option


class APositiveAValueEstimator(AValueEstimator):
    '''
    Estimator for the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes with magnitude m_i >= m_i-1 + dmc.

    Source:
        Following the idea of positivity of
        van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2).
        Note: This is *not* a-positive as defined by van der Elst and Page 2023
        (JGR: Solid Earth, Vol 128, Issue 10).
    '''

    weights_supported = False

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  times: np.ndarray,
                  mc: float,
                  delta_m: float,
                  dmc: float | None = None,
                  scaling_factor: float | None = None,
                  m_ref: float | None = None,
                  b_value: float | None = None,
                  *args,
                  weights: np.ndarray | list | None = None,
                  **kwargs) -> float:
        """
        Args:
            magnitudes: Array of magnitudes
            times: vector of times of the events, in any format (datetime,
                float, etc.)
            mc:         Completeness magnitude
            delta_m:    Discretization of magnitudes.
            dmc:        minimum magnitude difference between consecutive events.
                If None, the default value is delta_m.
            magnitudes: Array of magnitudes
            scaling_factor:     scaling factor.
                If given, this is used to normalize
                the number of observed events. For example:
                Volume or area of the region considered
                or length of the time interval, given in the unit of interest.
            m_ref:      reference magnitude for which the a-value is estimated.
            b_value:    b-value of the Gutenberg-Richter law. Only relevant
                when m_ref is not None.
            weights:    Array of weights for the magnitudes.
        """
        return super().calculate(magnitudes,
                                 times=times,
                                 mc=mc,
                                 delta_m=delta_m,
                                 dmc=dmc,
                                 scaling_factor=scaling_factor,
                                 m_ref=m_ref,
                                 b_value=b_value,
                                 weights=weights,
                                 *args,
                                 **kwargs)

    def _estimate(self,
                  dmc: float | None = None) -> float:
        if dmc is None:
            dmc = self.delta_m
        elif dmc < 0:
            raise ValueError("dmc must be larger or equal to 0")
        elif dmc < self.delta_m and get_option("warnings") is True:
            warnings.warn("dmc is smaller than delta_m, not recommended")

        # order the magnitudes and times
        idx = np.argsort(self.times)
        self.magnitudes = self.magnitudes[idx]
        times = self.times[idx]

        # differences
        mag_diffs = np.diff(self.magnitudes)
        time_diffs = np.diff(times)

        # only consider events with magnitude difference >= dmc
        idx = mag_diffs > dmc - self.delta_m / 2
        mag_diffs = mag_diffs[idx]
        time_diffs = time_diffs[idx]

        # estimate the number of events within the time interval
        total_time = times[-1] - times[0]

        time_factor = sum(time_diffs / total_time)
        n_pos = sum(idx) / time_factor

        # estimate a-value
        a = np.log10(n_pos)
        a = self._reference_scaling(a)
        return a
