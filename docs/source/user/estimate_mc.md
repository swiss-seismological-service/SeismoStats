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

The Maximum Curvature method (MAXC) defines the completeness threshold of a catalog as the magnitude at which the (non cumulative) FMD is maximal, given a binning of `fmd_bin`.

$$
m_c = \text{argmax}_{m_i} \left(N(m_i)\right) + \delta,
$$

where $N(m)$ is the number of earthquakes within a magnitude bin of width `fmd_bin` (not necessarily equal to the discretization of the catalog, $\Delta m$) and $\delta$ is the correction term which is set to avoid underestimation (by default set to +0.2). 

This method is based on the work of Wiemer & Wyss 2000 and Woessner & Wiemer 2005 and is implemented in the {func}`estimate_mc_maxc <seismostats.analysis.estimate_mc_maxc>` function.

```python
>>> from seismostats.analysis import estimate_mc_maxc

>>> cat = Catalog.from_dict({
    'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                    1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                    1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                    1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                    1.3, 1.1, 1.5, 1.4]})
>>> cat.estimate_mc_maxc(fmd_bin=0.1)
>>> cat.mc
1.4
```

The mc_maxc method also returns the correction factor used in the calculation of the best mc value.

```python
>>> best_mc, mc_info = cat.estimate_mc_maxc(fmd_bin=0.1)
>>> mc_info['correction_factor']
0.2
 ```

## K-S distance

The Kolmogorov-Smirnov (K-S) method estimates the magnitude of completeness Mc by comparing the observed and expected cumulative distribution functions (CDFs), assuming an exponential distribution of magnitudes. Mc is defined as the lowest magnitude where the simulated probability $P(D_{KS})$ of observing the KS-distance exceeds a chosen threshold $p$, with default values of $p$=0.1 and $n$=10,000 ($n$ being the number of random draws of discretized magnitudes drawn from an exponential distribution).

$$
m_c = \text{min}\left\{ m_i \mid p(D^i_{KS}) \geq p_{th} \right\},
$$

where $D^i_{KS}$ is the KS-distance of the observed CDF from the theoretical expected distribution when using $m_i$ as a lower cutoff, $p_{th}$ is the threshold that can be chosen freely, and $p(D_{KS})$ is the probability of a KS-distance equal or larger than the observed one under the assumption that the magnitudes follow the theoretical expected distribution.


This method is based on the work of Clauset et al., (2009) and Mizrahi et al., (2021), and implemented in the {func}`estimate_mc_ks <seismostats.analysis.estimate_mc_ks>` function.


```python
>>> from seismostats.analysis import estimate_mc_ks
>>> cat = Catalog.from_dict({
    'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                    1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                    1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                    1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                   1.3, 1.1, 1.5, 1.4]})
>>> cat.delta_m = 0.1
>>> cat.estimate_mc_ks()
>>> cat.mc
 1.0
```
The mc_ks method returns additional information about the calculation of the best mc, like b-values tested and ks distances. Those are returned by the method and can be used for further analysis.

```python
>>> best_mc, mc_info = cat.estimate_mc_ks()
>>> (mc_info['b_values_tested'], mc_info['ks_ds'])

([0.9571853220063774], [0.1700244200244202])
```

## Mc by b-value stability

The Mc by b-value stability method estimates the magnitude of completeness Mc by identifying where the b-value becomes stable as smaller magnitudes are excluded, indicating catalog completeness (assuming the magnitudes are exponentially distributed). It defines Mc as the lowest magnitude where the b-value variation across a range $L$ remains within its theoretical standard deviation ($\sigma_{b}$). 

$$
m_c = \text{min}\left\{ m_i \mid \text{abs} \left(\frac{1}{K} \sum_{k=1}^{K} b(m_i + k\cdot \Delta m^*) - b(m_i)\right) < \sigma_{b(m_i)} \right\},
$$

where $b(m)$ is the b-value estimate of all magnitudes above $m$, $\sigma_b$ is its uncertainty, $\Delta m^*$ is the magnitude bin size considered for $m_c$ precision and $K$ is the number of $m_c$ bins that is considered for stability. $L=K\cdot \Delta m^*$ is then the length of the magnitude range considered for stability. Note that in SeismoStats, the input required for b-value estimation is $L$, not $K$.

Users can specify the magnitude bin size $\Delta m$, with $L=0.5$ as the default stability range.


This method is based on the work of Cao & Gao 2002, and Woessner & Wiemer 2005 and is implemented in the {func}`estimate_mc_b_stability <seismostats.analysis.estimate_mc_b_stability>` function.

```python
>>> from seismostats.analysis import estimate_mc_b_stability

>>> cat = Catalog.from_dict({
            'magnitude': [2.3, 1.2, 1.5, 1.2, 1.7, 1.1, 1.2, 1.5,
                         1.8, 1.6, 1.2, 1.5, 1.2, 1.7, 1.6, 1.1,
                         1.1, 1.2, 2.0, 1.1, 1.2, 1.1, 1.2, 1.6,
                         1.9, 1.3, 1.7, 1.3, 1.0, 1.2, 1.7, 1.3,
                         1.3, 1.1, 1.5, 1.4]})
>>> cat.delta_m = 0.1
>>> cat.estimate_mc_b_stability()
>>> cat.mc
1.1
```
The mc_b_stability method returns additional information about the calculation of the best mc, like b-values tested and the array of differences. Those are returned by the method and can be used for further analysis.

```python
 >>> best_mc, mc_info = cat.estimate_mc_b_stability()
>>> (mc_info['mcs_tested'], mc_info['diff_bs'])
(array([1. , 1.1]), [2.23375277112158, 0.9457747650207577])
```

## References
- Cao, A., & Gao, S. S. (2002). Temporal variation of seismic b-values beneath northeastern Japan island arc. Geophysical Research Letters, 29(9), 1–3. https://doi.org/10.1029/2001gl013775

- Clauset, A., Shalizi, C.R. and Newman, M.E., 2009. Power-law distributions in empirical data. SIAM review, 51(4), pp.661-703.

- Mizrahi, L., Nandan, S. and Wiemer, S., 2021. The effect of declustering on the size distribution of mainshocks. Seismological Society of America, 92(4), pp.2333-2342.

- Wiemer, S. and Wyss, M., 2000. Minimum magnitude of completeness in earthquake catalogs: Examples from Alaska, the western United States, and Japan. Bulletin of the Seismological Society of America, 90(4), pp.859-869.

- Woessner, J. and Wiemer, S., 2005. Assessing the quality of earthquake catalogues: Estimating the magnitude of completeness and its uncertainty. Bulletin of the Seismological Society of America, 95(2), pp.684-698.       