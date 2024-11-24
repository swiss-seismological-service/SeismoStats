# %%
from seismostats.analysis import (estimate_b_more_positive,
                                  estimate_b_positive, estimate_b_utsu)
from seismostats.analysis.bvalue import (BMorePositiveEstimator,
                                         BPositiveEstimator,
                                         UtsuBValueEstimator)
from seismostats.utils.simulate_distributions import simulate_magnitudes_binned

mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)
mc = 0
delta_m = 0.1

# %%
b = estimate_b_more_positive(magnitudes=mags, delta_m=delta_m)
b

# %%
estimator = BMorePositiveEstimator(mc=mc, delta_m=delta_m)
# estimate beta value
estimator(mags)

# %%
mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

b = estimate_b_positive(magnitudes=mags, delta_m=delta_m)
b

# %%
estimator = BPositiveEstimator(mc=mc, delta_m=delta_m)
# estimate beta value
estimator(mags)
# %%
mags = simulate_magnitudes_binned(n=100, b=1, mc=0, delta_m=0.1)

b = estimate_b_utsu(magnitudes=mags, delta_m=delta_m, mc=mc)
b

# %%
estimator = UtsuBValueEstimator(mc=mc, delta_m=delta_m)
# estimate beta value
estimator(mags)

# %%
