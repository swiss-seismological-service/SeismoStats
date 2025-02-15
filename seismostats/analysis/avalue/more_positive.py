import warnings

import numpy as np

from seismostats.analysis.avalue.base import AValueEstimator
from seismostats.analysis.bvalue.utils import find_next_larger
from seismostats.utils._config import get_option


class AMorePositiveAValueEstimator(AValueEstimator):
    '''
    Return the a-value of the Gutenberg-Richter (GR) law using only the
    earthquakes with magnitude m_i >= m_i-1 + dmc.

    Source:
        van der Elst and Page 2023 (JGR: Solid Earth, Vol 128, Issue 10).
    '''

    def __init__(self):
        super().__init__()

    def calculate(self,
                  magnitudes: np.ndarray,
                  mc: float,
                  delta_m: float,
                  times: np.ndarray,
                  b_value: float,
                  scaling_factor: float | None = None,
                  m_ref: float | None = None,
                  dmc: float | None = None,
                  ) -> float:
        '''
        Args:
            magnitudes:     Array of magnitudes
            mc:             Completeness magnitude
            delta_m:        Discretization of magnitudes
            times:          Vector of times of the events, in any format
                            (datetime, float, etc.).
            b_value:        B-value of the Gutenberg-Richter law.
            scaling_factor: Scaling factor
                            If given, this is used to normalize the number of
                            observed events. For example: Volume or area of the
                            region considered or length of the time interval,
                            given in the unit of interest.
            m_ref:          Reference magnitude for which the a-value
                            is estimated.
            dmc:            Minimum magnitude difference between consecutive
                            events. If None, the default value is delta_m.

        Returns:
            a_pos: a-value of the Gutenberg-Richter distribution
        '''

        if not b_value:
            # if using estimate_a function, b_value can be None even though its
            # not passed to the function
            raise ValueError("b_value must be given")

        self.times: np.ndarray = np.array(times)
        self.dmc: float = dmc if dmc is not None else delta_m

        if self.dmc < 0:
            raise ValueError("dmc must be larger or equal to 0.")

        if self.dmc < delta_m and get_option("warnings") is True:
            warnings.warn("dmc is smaller than delta_m, not recommended.")

        return super().calculate(magnitudes,
                                 mc=mc,
                                 delta_m=delta_m,
                                 scaling_factor=scaling_factor,
                                 m_ref=m_ref,
                                 b_value=b_value,
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

        # find next larger event (if it exists)
        idx_next_larger = find_next_larger(
            self.magnitudes, self.delta_m, self.dmc)
        time_diffs = self.times[idx_next_larger] - self.times

        # deal with events which do not have a next larger event
        idx_no_next = idx_next_larger == 0
        time_diffs[idx_no_next] = self.times[-1] - self.times[idx_no_next]

        # estimate the number of events within the time interval
        total_time = self.times[-1] - self.times[0]

        # scale the time
        tau = time_diffs * 10**(-self.b_value
                                * (self.magnitudes + self.dmc - self.mc))

        time_factor = sum(tau / total_time)
        n_more_pos = sum(~idx_no_next) / time_factor

        # make sure that all attributes are consistent
        idx_next_larger = idx_next_larger[~idx_no_next]
        self.magnitudes = self.magnitudes[idx_next_larger]
        self.times = self.times[idx_next_larger]
        self.idx = self.idx[idx_next_larger]

        # estimate a-value
        return np.log10(n_more_pos)
