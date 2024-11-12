import numpy as np

from seismostats.analysis.b_values.classic import ClassicBValueEstimator


def estimate_b(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0,
    weights: list | None = None,
    b_parameter: str = 'b_value',
    return_std: bool = False,
    method=ClassicBValueEstimator,
    return_n: bool = False,
    *args,
    **kwargs
) -> float | tuple[float, float] | tuple[float, float, float]:

    estimator = method(mc=mc, delta_m=delta_m, *args, **kwargs)

    if b_parameter == 'b_value':
        b = estimator(magnitudes, weights=weights)
    elif b_parameter == 'beta':
        b = estimator.estimate_beta(magnitudes, weights=weights)
    else:
        raise ValueError('b_parameter must be either "b_value" or "beta"')

    out = b

    if return_std:
        out = (out, estimator.std)

    if return_n:
        out = (*tuple(out), estimator.n)

    return out
