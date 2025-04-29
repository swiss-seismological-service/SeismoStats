# Estimate magnitude of completeness
```{eval-rst}
.. currentmodule:: seismostats
```

The magnitude of completeness represent the magnitude from which all events are assumed to be detected for a given network and time period.

Currently, `SeismoStats` supports three methods to estimate the magnitude of completeness:
1. **Maximum Curvature** (MAXC)
2. **K-S distance** (KS)
3. **Mc by b-value stability** (MBS)

## Maximum Curvature

The Maximum Curvature method (MAXC) defines the completeness threshold of a catalog as the magnitude at which the (non cumulative) FMD is maximal.

$$
m_c = \argmax_m (N(m_i < m \leq m_{i} + \Delta m)) + \delta
$$

where $$N(m)$$ is the number of earthquakes within a magnitude bin of width $$\Delta m$$, and $$\delta$$ is the correction factor which is set to avoid underestimation. 

This method is based on the work of Wiemer & Wyss 2000 and Woessner & Wiemer 2005 and is implemented in the {func}`estimate_mc_maxc <seismostats.analysis.estimate_mc_maxc>` function.

```python
from seismostats.analysis import estimate_mc_maxc

mc = estimate_mc_maxc(cat.magnitude, delta_m=0.1)
print(f'Maximum curvature method: Mc = {mc:.1f}')
```
### References
- Wiemer, S. and Wyss, M., 2000. Minimum magnitude of completeness in earthquake catalogs: Examples from Alaska, the western United States, and Japan. Bulletin of the Seismological Society of America, 90(4), pp.859-869.

- Woessner, J. and Wiemer, S., 2005. Assessing the quality of earthquake catalogues: Estimating the magnitude of completeness and its uncertainty. Bulletin of the Seismological Society of America, 95(2), pp.684-698.

## K-S distance
