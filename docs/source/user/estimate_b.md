# Estimate b-value

The b-value is the parameter in the Gutenberg-Richter law that quantifies the relative frequency of small earthquakes vs. large earthquakes. A b-value of 1 means that for every increase of one unit in magnitude, the number of earthquakes decreases by a factor of 10. A smaller b-value translates to relatively more large erarthquakes, and vice versa.

Here, we
1. Explain the basics of b-value estimation, and 
2. Show the different methods that b-value estimation can be done with SeismoStats.

## 1. Short Introduction on b-value Estimation
The GR law above the completeness magnitude $m_c$ can be expressed as

$$
N(m) = 10^{a - b (m - m_c)},
$$

where $N(m)$ is the number of events with magnitudes larger than or equal to $m$ in a given catalog, and $a$ and $b$ are the a-value and b-value, respectively. This relation implies that magnitudes follow an exponential distribution. In the case of continuous magnitudes (i.e., no discretization), the maximum likelihood estimator of the b-value (Aki, 1965), $\hat{b}$, is given as

$$
\hat{b} = \frac{\log e}{\frac{1}{n} \sum_{i=1}^n m_i - m_c},  \tag{1}
$$

where $m_1, \dots m_n$ are the individual magnitudes above $m_c$ within the catalog. The uncertainty $\sigma(\hat{b})$ of the estimate, which stems from the fact that the magnitude sample is finite, is given as (Shi and Bolt, 1892)

$$
\sigma(\hat{b}) = \frac{\ln(10) \cdot \hat{b}^2}{\sqrt{n-1}} \cdot \sqrt{\text{var}(m)},  \tag{2}
$$

where $\text{var }(m)$ is the variance of the magnitudes which can be estimated from the data.

Below we discuss shortly the effect of discretized magnitudes, weighted magnitudes, and two alternative methods of b-value estimation that reduce the effect of short term aftershock incompleteness.

### 1.1 Discretization of Magnitudes
In most cases, the magitudes given in a catalog are discretized. Utsu (1966) gave an approximative formulation of the b-value estimator for this case, effectively by shifting the magnitudes by half a bin. However, the correct (unbiased) estimate takes a slightly more complicated form (see Tinti and Mulargia, 1987), which we implemented for all b-value estimators available in SeismoStats (except the one that uses Utsu's formula). For all esimtators, $\Delta m$, the bin size of discretized magnitudes, is a mandatory input. It can be set to zero if no discretization is present. For most catalogs of natural seismicity, however, $\Delta m$ is between $0.01$ and $0.2$. In our implementation, the uncertainty estimate (Eq. 2) is not adjusted for the binning, but synthetic tests suggest that the differences are negligible.

### 1.2 Weighted Magnitudes
Most b-value estimators implemented in this package support weighted magnitudes. This can be useful when magnitudes should contribute unequally to the b-value estimate—for instance, based on their distance from a location of interest, or their likelihood of being related to a specific phenomenon.

### 1.3 Positive Methods

#### 1.3.1 b-positive
In an effort to reduce the impact of short term incompleteness, Van der Elst (2021) introduced the b-positive estimator. This method assumes that the detection threshold at each point in time is given by the magnitude of the most recent earthquake, plus a threshold value $\delta m_c$.

The estimation is implemented as follows. First, the magnitude differences between consecutive event pairs ($m_i - m_{i-1}$) are calculated. Only the differences that are larger than $\delta m_c$ (and therefore positive) are used for b-value estimation. The b-value can be estimated from the positive differences using the maximum likelihood estimator. This relies on the fact that differences between values drawn from an exponential distribution follow a Laplace (or double exponential) distribution.

Although the b-positive method does effectively eliminate the effect of short-term aftershock incompleteness, it remains sensitive to baseline catalog incompleteness due to network limitations [citation needed]. This effect can be reduced either by only considering the complete catalog (above $m_c$) before estimating the differences or by increasing the threshold $\delta m_c$. In the SeismoStats package, both the completeness magnitude ($m_c$) and threshold ($\delta m_c$) can be given as an input, with the default being $\delta m_c = \Delta m$.

#### 1.3.2 b-more-positive.
Based on the b-positive method, Lippiello and Petrillo (2024) developed the b-more-positive method, which makes use of the same principles as b-positive. For this method, the magnitude differences are constructed in an alternative way. For each earthquake, the next earthquake with a magnitude that is larger by at leas $\delta m_c$ is used to calculate the differences. Therefore, the number of differences is larger for this method than the b-positive method, which typically halves the data size. However, the variance of the estimate is not reduced as a result of using the b-more-positive method. Since the uncertainty estimate in Eq. (2) depends on the number of input magnitudes, it underestimates the true uncertainty in this case — typically by a factor of two.

## 2. Estimation of the b-value
In SeismoStats, we provide several ways to estimate the b-value:
- Using the {ref}`BValueEstimator <reference/analysis/bvalues:Estimators>` class
- Using the function {func}`estimate_b <seismostats.analysis.estimate_b>` (this is the easiest way, {ref}`jump here </user/estimate_b.md#estimate-b>`)
- Using the method {func}`estimate_b <seismostats.Catalog.estimate_b>` native to the Catalog class (most practical if the catalog format is used, {ref}`jump here </user/estimate_b.md#cat-estimate-b>`)

Below, we show examples for each method.

### 2.1 BValueEstimator
All b-value estimations in SeismoStats are built upon the {ref}`BValueEstimator <reference/analysis/bvalues:Estimators>` class, which defines a unified interface for different estimation methods. It requires the following inputs: an array of magnitudes $m_1, \dots, m_n$, the magnitude of completeness $m_c$, and the magnitude discretization $\Delta m$. This base class is then extended to implement specific estimation techniques. Currently, four methods are available: {class}`ClassicBValueEstimator <seismostats.analysis.ClassicBValueEstimator>`, {class}`BPositiveBValueEstimator <seismostats.analysis.BPositiveBValueEstimator>`, {class}`BMorePositiveBValueEstimator <seismostats.analysis.BMorePositiveBValueEstimator>`, and {class}`UtsuBValueEstimator <seismostats.analysis.UtsuBValueEstimator>`.

The class can be used as follows:

```python
>>> from seismostats.analysis import ClassicBValueEstimator
>>> estimator = ClassicBValueEstimator()
>>> estimator.calculate(mags, mc, delta_m)
1.05
>>> estimator.b_value
1.05
```

When calling `.calculate()`, the estimator automatically excludes all magnitudes below the completeness threshold $m_c$. This behavior is consistent across all b-value estimators in SeismoStats, so it is essential to provide an accurate value for $m_c$.

Another crucial input is the magnitude discretization bin width $\Delta m$. For the positive methods, one can also specify a threshold $\delta m_c$ and an array of event times. By default, $\delta m_c$ is set equal to $\Delta m$. If no times are provided, the estimator assumes that the magnitudes are already ordered in time. 

The computed b-value is stored in the estimator instance (in this example, `estimator`) and can be accessed directly after calling `.calculate()` 

```python
>>> from seismostats.analysis import BPositiveBValueEstimator
>>> estimator = BPositiveBValueEstimator()
>>> estimator.calculate(mags, mc, delta_m, times=times, dmc=dmc)
0.98
>>> estimator.b_value
0.98
```

Note that for {class}`BPositiveBValueEstimator <seismostats.analysis.BPositiveBValueEstimator>` and  {class}`BMorePositiveBValueEstimator <seismostats.analysis.BMorePositiveBValueEstimator>`, the parameter `mc` still is used in the same way as in the classical case: all magnitudes below $m_c$ are excluded from the analysis.

### 2.2 estimate_b
An alternative way to calculate a b-value is using the function {func}`estimate_b <seismostats.analysis.estimate_b>`. To estimate the b-value using Eq. (1), it only requires an array of magnitudes $m_1, \dots, m_n$, the magnitude of completeness $m_c$ and the discretization of magnitudes $\Delta m$.

```python
>>> from seismostats.analysis import estimate_b
>>> magnitudes = [0, 0, 1, 1, 1, 2, 3, 2, 3, 5, 6, 7]
>>> estimate_b(magnitudes, mc=1, delta_m=1)
0.169
```

Note that the function {func}`estimate_b <seismostats.analysis.estimate_b>` automatically disregards magnitudes below $m_c$. Therefore, it is crucial to provide the correct $m_c$. The default b-value estimation method used by `estimate_b()` is the classical method ({class}`ClassicBValueEstimator <seismostats.analysis.ClassicBValueEstimator>`). However, it is also possible to specify which method should be used. This can be done as follows:

```python
>>> from seismostats.analysis import estimate_b, BPositiveBValueEstimator
>>> times = numpy.arange(10)
>>> estimate_b(magnitudes, mc=1, delta_m=1, method=BPositiveBValueEstimator)
0.845
```

### 2.3 cat.estimate_b()
If you have already converted your data into a Catalog object, you can directly estimate the b-value using the internal method of the Catalog class, which functions just like the standalone `estimate_b()` function shown above.

```python
>>> estimator = cat.estimate_b(mc=1, delta_m=0.1)
>>> cat.b_value
0.882
```

If $\Delta m$ and $m_c$ are already defined for the catalog, you can omit them in the method call, and the stored values will be used:

```python
>>> cat.mc = 1
>>> cat.delta_m = 0.1
>>> estimator = cat.estimate_b()
>>> cat.b_value
0.882
```

This is especially convenient because both `mc` and `delta_m` are set typically set using the the `bin_magnitudes` method and the `estimate_mc` methods. 
```python
>>> # First, estimate mc
>>>cat.estimate_mc_maxc()
>>> # Now, it is set as an attribute 
>>> cat.mc
1.0
>>> # Second, bin the magnitudes
>>> cat.bin_magnitudes(delta_m=0.1, inplace=True)
>>> cat.delta_m
0.1
>>> estimator = cat.estimate_b()
>>> cat.b_value
0.882
```

## References
- Aki, Keiiti. "Maximum likelihood estimate of b in the formula log N= a-bM and its confidence limits." Bull. Earthquake Res. Inst., Tokyo Univ. 43 (1965): 237-239.
- Utsu, Tokuji. "A statistical significance test of the difference in b-value between two earthquake groups." Journal of Physics of the Earth 14.2 (1966): 37-40.
- Tinti, Stefano, and Francesco Mulargia. "Confidence intervals of b values for grouped magnitudes." Bulletin of the Seismological Society of America 77.6 (1987): 2125-2134.
- Shi, Yaolin, and Bruce A. Bolt. "The standard error of the magnitude-frequency b value." Bulletin of the Seismological Society of America 72.5 (1982): 1677-1687.
- van der Elst, Nicholas J. "B‐positive: A robust estimator of aftershock magnitude distribution in transiently incomplete catalogs." Journal of Geophysical Research: Solid Earth 126.2 (2021): e2020JB021027.
- Lippiello, E., and G. Petrillo. "b‐more‐incomplete and b‐more‐positive: Insights on a robust estimator of magnitude distribution." Journal of Geophysical Research: Solid Earth 129.2 (2024): e2023JB027849.