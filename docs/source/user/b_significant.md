# b-significant

Here, we show how to apply the one-dimanesional case of the b-significant method, as described in Mirwald et al. (2024). The idea is to test if the variation of the b-value is significant or not.

Here are the steps that have to be carried out in order to estimate the significancwe of b-value variation:
1. Order the magnitudes with respect to some other parameter. This could be the time, but also some other quantity as e.g. the depth or distance to some reference location
2. Evaluate the magnitude of completeness.  `mc` can be either given as a constant or as a vector of the same length as the magnitudes. 
3. Decide the number of magitudes used for each b-value estimation (`n_m`). The equal $n_m$ technique is appleid, that is, the b-valiue is estimated from a sliding window of `n_m` magnitudes.
4. Choose a b-value estimation method, e.g, `BPositiveBValueEstimator` (in this case, it is important to provide the `times` vector, too, unless the magnitudes are already sorted in time)

```python
>>> from seismostats.analysis import b_significant_1D
>>> mc = 1
>>> delta_m = 0.1
>>> n_m = 100
>>> p, mac, mu_mac, std_mac = b_significant_1D(mags, mc, delta_m, n_m)
```

The output of this function is the p-value connected to the null-hypothesis of a constant b-value (`p`), the mean autocorrelation (`mac`) as described by Mirwald et al. (2024), and the expected value (`mu_mac`) and standard deviation (`std_mac`) of the mean autocorrelation under the null hypothesis. If you found the variation to be significant (i.e., the p-value is below a chosen threshold, often set to 0.05), it is reasonable to analize how it is changing. For this, there exists also a plotting function.

```python
from seismostats.plots import plot_b_series_constant_nm
>>> mc = 1          # magnitude of completeness
>>> delta_m = 0.1   # binning of the magnitudes
>>> n_m = 100
>>> ax = plot_b_series_constant_nm(mags, delta_m, mc, times, n_m=n_m,x_variable=times, color='#1f77b4', plot_technique='right', label='classical b-value')

ax = plot_b_series_constant_nm(mags[idx], delta_m, mc, times[idx], n_m=n_m,x_variable=times[idx], color='red', plot_technique='right', label='b-positive', ax=ax, b_method=BPositiveBValueEstimator)

_ = plt.xticks(rotation=45)
ax.set_xlabel('Time')
```

## References
- Mirwald, Aron, Leila Mizrahi, and Stefan Wiemer. "How to b‐significant when analyzing b‐value variations." Seismological Research Letters 95.6 (2024)