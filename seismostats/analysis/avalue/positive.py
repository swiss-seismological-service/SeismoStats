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
                  dmc: float | None = None,
                  correction: bool = False,
                  *args,
                  **kwargs) -> float:
        """
        Args:
            magnitudes: Array of magnitudes
            times: vector of times of the events, in any format (datetime,
                float, etc.)
            dmc:        minimum magnitude difference between consecutive events.
                If None, the default value is delta_m.
            correction: If True,
                the a-value is corrected for the bias introduced
                by the observation period being larger than the time interval
                between the first and last event. This is only relevant if the
                sample is very small (<30). The assumption is that the blind
                time in the beginning and end of the catalogue has a similar
                distribution as the rest of the catalogue. We think that this
                improves the estimate of the a-value for small samples, however,
                without proof.
        """
        return super().calculate(magnitudes,
                                 times=times,
                                 dmc=dmc,
                                 correction=correction,
                                 *args,
                                 **kwargs)

    def _estimate(self,
                  times: np.ndarray,
                  dmc: float | None = None,
                  correction: bool = False) -> float:

        if self.mc is None:
            self.mc = self.magnitudes.min()
        elif self.magnitudes.min() < self.mc:
            if get_option("warnings") is True:
                warnings.warn(
                    "Completeness magnitude is higher"
                    "than the lowest magnitude."
                    "Cutting the magnitudes to the completeness magnitude.")
            idx = self.magnitudes >= self.mc - self.delta_m / 2
            self.magnitudes = self.magnitudes[idx]
            times = times[idx]
        if dmc is None:
            dmc = self.delta_m
        elif dmc < 0:
            raise ValueError("dmc must be larger or equal to 0")
        elif dmc < self.delta_m and get_option("warnings") is True:
            warnings.warn("dmc is smaller than delta_m, not recommended")

        # order the magnitudes and times
        idx = np.argsort(times)
        self.magnitudes = self.magnitudes[idx]
        times = times[idx]

        # differences
        mag_diffs = np.diff(self.magnitudes)
        time_diffs = np.diff(times)

        # only consider events with magnitude difference >= dmc
        idx = mag_diffs > dmc - self.delta_m / 2
        mag_diffs = mag_diffs[idx]
        time_diffs = time_diffs[idx]

        # estimate the number of events within the time interval
        total_time = times[-1] - times[0]
        total_time_pos = sum(time_diffs / total_time)
        if correction:
            total_time += 2 * np.mean(np.diff(times) / total_time)
            total_time_pos += np.mean(time_diffs / total_time)
        n_pos = 1 / total_time_pos * len(mag_diffs)

        # estimate a-value
        a = np.log10(n_pos)
        a = self._reference_scaling(a)
        return a
