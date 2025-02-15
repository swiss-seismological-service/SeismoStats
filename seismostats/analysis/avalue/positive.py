import warnings

import numpy as np

from seismostats.analysis.avalue.base import AValueEstimator
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

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  times: np.ndarray,
                  scaling_factor: float | None = None,
                  m_ref: float | None = None,
                  b_value: float | None = None,
                  dmc: float | None = None,
                  ) -> float:
        '''
        Args:
            magnitudes:     Array of magnitudes
            mc:             Completeness magnitude
            delta_m:        Discretization of magnitudes
            times:          Vector of times of the events, in any format
                            (datetime, float, etc.).
            scaling_factor: Scaling factor
                            If given, this is used to normalize the number of
                            observed events. For example: Volume or area of the
                            region considered or length of the time interval,
                            given in the unit of interest.
            m_ref:          Reference magnitude for which the a-value
                            is estimated.
            b_value:        B-value of the Gutenberg-Richter law. Only relevant
                            when m_ref is not None.
            dmc:            Minimum magnitude difference between consecutive
                            events. If None, the default value is delta_m.

        Returns:
            a_pos: a-value of the Gutenberg-Richter distribution
        '''

        self.times: np.ndarray = np.array(times)
        self.dmc: float = dmc if dmc is not None else delta_m

        if self.dmc < 0:
            raise ValueError('dmc must be larger or equal to 0.')

        if self.dmc < delta_m and get_option('warnings') is True:
            warnings.warn('dmc is smaller than delta_m, not recommended.')

        return super().calculate(magnitudes,
                                 mc=mc,
                                 delta_m=delta_m,
                                 scaling_factor=scaling_factor,
                                 m_ref=m_ref,
                                 b_value=b_value
                                 )

    def _filter_magnitudes(self) -> np.ndarray:
        '''
        Filter out magnitudes below the completeness magnitude.
        '''
        super()._filter_magnitudes()
        self.times = self.times[self.idx]
        return self.idx

    def _estimate(self) -> float:
        # order the magnitudes and times
        srt = np.argsort(self.times)
        self.magnitudes = self.magnitudes[srt]
        self.times = self.times[srt]
        self.idx = self.idx[srt]

        # differences
        mag_diffs = np.diff(self.magnitudes)
        time_diffs = np.diff(self.times)

        # only consider events with magnitude difference >= dmc
        is_larger = mag_diffs >= self.dmc - self.delta_m / 2
        time_diffs = time_diffs[is_larger]
        self.magnitudes = self.magnitudes[1:][is_larger]
        self.idx = self.idx[1:][is_larger]

        # estimate the number of events within the time interval
        total_time = self.times[-1] - self.times[0]
        self.times = self.times[1:][is_larger]

        time_factor = sum(time_diffs / total_time)
        n_pos = sum(is_larger) / time_factor

        # estimate a-value
        return np.log10(n_pos)
