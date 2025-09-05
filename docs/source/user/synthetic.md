# Synthetic magnitude generation

Sometimes it is needed to create synthetic earhtquake magnitudes. Here, we describe briefly the way generate magnitudes drawn from an exponential distribution with a given discretization and completeness. 


```python
>>> from seismostats.utils import simulate_magnitudes_binned
>>> from seismostats.plots import plot_mags_in_time
>>> n = 100         # number of events to simulate
>>> b_value = 1.5   # b-value of the synthetic catalog
>>> delta_m = 0.05  # magnitude binning of the synthetic catalog
>>> mc = np.ones(n) # completeness of the synthetic catalog
>>> mc[n//2:] = 1
>>> mags = simulate_magnitudes_binned(n, b_value, mc, delta_m)
>>> ax = plot_mags_in_time(mags, np.arange(len(mags)))
```

Some notes on the implementation:
1. The magnitudes are first drawn from an exponential distribution (with the b-value provided), and then discretized with `delta_m`. If no `delta_m` is given, the magnitudes are continuous (down to the float precision)
2. It is possible to vary the completeness and b-value for each earthquake. In the example below, we did so for the parameter `mc`. If this is not desired, just `mc` can be given as a simple float.
3. It is also possible to generate magnitudes from a truncated exponential distribution. For this, privide the parameter `mag_max`.

<figure>
  <img src="../_static/synthetic_mags.png" alt="Alt text" width="600"/>
  <figcaption>Figure 1: Synthetic magnitudes with a b-value of 1.5 and changing completeness.</figcaption>
</figure>