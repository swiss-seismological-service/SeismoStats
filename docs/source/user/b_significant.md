# b-significant

Here, we show how to apply the one-dimanesional case of the b-significant method, as described in Mirwald et al. (2024). The idea is to test if the variation of the b-value is significant or not.

## 1. How to use b_significant_1D

Here are the steps that have to be carried out in order to estimate the significancwe of b-value variation:
1. Order the magnitudes with respect to some other parameter. Alternatively, provide the values of the dimension of interest (`x_variable`)
2. Evaluate the magnitude of completeness.  `mc` can be either given as a constant or as a vector of the same length as the magnitudes. 
3. Decide the number of magitudes used for each b-value estimation (`n_m`). The equal $n_m$ technique is appleid, that is, the b-valiue is estimated from a sliding window of `n_m` magnitudes. In order for the b-significant method to work (expecially the normality assumption), it should be at least 20 and at most $n/15$, where  $n$ is the number of earthquakes.
4. Choose a b-value estimation method, e.g, `BPositiveBValueEstimator`

```python
>>> from seismostats.analysis import b_significant_1D, BPositiveBValueEstimator
>>> mc = 1
>>> delta_m = 0.1
>>> n_m = 100
>>> p, mac, mu_mac, std_mac = b_significant_1D(mags, mc, delta_m, times, n_m, x_variable=times, method= BPositiveBValueEstimator)
>>> p
0.01
```

The output of this function is the p-value connected to the null-hypothesis of a constant b-value (`p`), the mean autocorrelation (`mac`) as described by Mirwald et al. (2024), and the expected value (`mu_mac`) and standard deviation (`std_mac`) of the mean autocorrelation under the null hypothesis. If you found the variation to be significant (i.e., the p-value is below a chosen threshold, often set to 0.05), it is reasonable to analize how it is changing. For this, there exists also a plotting function.

## 2. Plot the b-value series with a certain `n_m`

```python
>>> from seismostats.plots import plot_b_series_constant_nm
>>> mc = 1
>>> delta_m = 0.1
>>> n_m = 100
>>> ax = plot_b_series_constant_nm(mags, delta_m, mc, times, n_m=n_m, x_variable=times, color='red', plot_technique='right', label='b-positive', ax=ax, method=BPositiveBValueEstimator)
```

<figure>
  <img src="../_static/b_value_timeseries_0.png" alt="Alt text" width="600"/>
  <figcaption>Figure 1: b-value time series with n_m = 100.</figcaption>
</figure>


## 3. Visualize the autocorrelation of b-values using different `n_m`

In case that it is aleady clear that the b0-value is varying, but you need to find out the length (or time) scale at which the variation is strongest, you can simply apply the b-positive method with different values of `n_m`. If you want to do this visually, we also have implemented a function for this.

```python
>>> from seismostats.plots import plot_b_series_constant_nm
>>> mc = 1
>>> delta_m = 0.1
>>> n_m = 100
>>> ax = plot_b_series_constant_nm(mags, delta_m, mc, times, n_m=n_m, x_variable=times, color='red', plot_technique='right', label='b-positive', ax=ax, method=BPositiveBValueEstimator)
```

<figure>
  <img src="../_static/b_significant_0.png" alt="Alt text" width="600"/>
  <figcaption>Figure 2: b-significant method applied for different n_m
  . If the value is outside of the shaded area, the b-value variation can be jugded to be significant. </figcaption>
</figure>

Note that this plot follows the convention that $n_m$ is the total number of earthquakes above the completeness that is used as input for the b-value estimation. However, when testing if the number of magnitudes is above `min_num` (which is per default set as 20), the effective number of magnitudes is used instead. If less than `min_num` magnitudes are present, the method returns NaNs.  For the b-positive method, the effective number is around half the number of original events, therefore it only shows results for $n_m > 40$.

## References
- Mirwald, Aron, Leila Mizrahi, and Stefan Wiemer. "How to b‐significant when analyzing b‐value variations." Seismological Research Letters 95.6 (2024)