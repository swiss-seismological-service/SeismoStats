from typing import Literal

import numpy as np

from seismostats.analysis.bvalue.base import BValueEstimator
from seismostats.analysis.bvalue.classic import ClassicBValueEstimator
from seismostats.analysis.bvalue.more_positive import \
    BMorePositiveBValueEstimator  # noqa
from seismostats.analysis.bvalue.positive import \
    BPositiveBValueEstimator  # noqa
from seismostats.analysis.bvalue.utils import b_value_to_beta  # noqa
from seismostats.analysis.bvalue.utils import beta_to_b_value  # noqa
from seismostats.analysis.bvalue.utils import shi_bolt_confidence  # noqa
from seismostats.analysis.bvalue.utsu import UtsuBValueEstimator  # noqa


def estimate_b(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float,
    weights: list | None = None,
    b_parameter: Literal['b_value', 'beta'] = 'b_value',
    return_std: bool = False,
    method: BValueEstimator = ClassicBValueEstimator,
    return_n: bool = False,
    **kwargs
) -> float | tuple[float, float] | tuple[float, float, float]:
    '''
        Returns the b-value of the Gutenberg-Richter (GR) law.

        .. math::
            N(m) = 10 ^ {a - b \\cdot (m - m_{ref})},

        where :math:`N(m)` is the number of events with magnitude larger than
        or equal to :math:`m` that occurred in the timeframe of the catalog,
        :math:`a` and :math:`b` are the a- and b-value, and :math:`m_{ref}`
        is the reference magnitude above which earthquakes are counted.

        Args:
            magnitudes:     Array of magnitudes.
            mc:             Completeness magnitude.
            delta_m:        Bin size of discretized magnitudes.
            weights:        Array of weights for the magnitudes.
            b_parameter:    If 'b_value', the b-value is returned. If 'beta',
                        the beta-value is returned. beta is the analogous value
                        to b, but when formulating the GR law with base e.
            return_std:     If True, the standard deviation of the b-value is
                        also returned.
            method:         BValueEstimator class to use for calculation.
            return_n:       If True, the number of events used for the
                        estimation is also returned.
            **kwargs:       Additional parameters to be passed to the
                        :func:`calculate` method.

        Returns:
            b:      b-value or beta of the Gutenberg-Richter law.
            std:    Standard deviation of the b-value or beta. Only returned if
                `return_std` is True.
            n:      Number of events used for the estimation. Only returned if
                `return_n` is True.

        Examples:
            .. code-block:: python

                >>> import numpy as np
                >>> from seismostats.analysis import estimate_b

                >>> magnitudes = np.array([2. , 2.5, 2.1, 2.2, 2.5, 2.2, 2.6,
                ...                        2.3, 2.7, 2.2, 2.4, 2. , 2.7, 2.2,
                ...                        2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7,
                ...                        2.4, 2.2, 2.5])
                >>> mc = 2.0
                >>> delta_m = 0.1

                >>> b = estimate_b(magnitudes=magnitudes,
                ...                mc=mc,
                ...                delta_m=delta_m)
                >>> b

                1.114920128810535

            .. code-block:: python

                >>> from seismostats.analysis import BPositiveBValueEstimator

                >>> times = np.arange(len(magnitudes))
                >>> b = estimate_b(magnitudes,
                ...                mc,
                ...                delta_m,
                ...                times=times,
                ...                method=BPositiveBValueEstimator)
                >>> b

                1.5490195998574323
        '''

    estimator = method()
    estimator.calculate(magnitudes, mc=mc, delta_m=delta_m,
                        weights=weights, **kwargs)

    if b_parameter == 'b_value':
        b = estimator.b_value
    elif b_parameter == 'beta':
        b = estimator.beta
    else:
        raise ValueError('b_parameter must be either "b_value" or "beta"')

    out = b

    if return_std:
        if b_parameter == 'b_value':
            out = (out, estimator.std)
        elif b_parameter == 'beta':
            out = (out, estimator.std_beta)

    if return_n:
        out = (*tuple(out), estimator.n)

    return out
