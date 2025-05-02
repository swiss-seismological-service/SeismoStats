# Estimate b-value

The b-value is the parameter in the Gutenberg-Richter law that contains information on the relative share of small earthquakes vs. large earthquakes. A b-value of one means that there is a reduction of frequency of 10 if the magnitude goes up by one. A smaller b-value translates to relatively more large erarthquakes, and vice versa.

Here, we will
1. Explain the basics of b-value estimation, and 
2. Show the different methods that b-value estimation can be done with SeismoStats.

## 1. Short Introduction on b-value Estimation
The GR law above the completeness magnitude $m_c$ can be expressed as follows:

$$
N(m) = 10^{a - b (m - m_c)},
$$

where $N(m)$ is the number of events with magnitudes larger than or equal to $m$ that occurred in the catalog. This relation is equivalent to the magnitudes being distributed exponentially. In the case of continuous magnitudes (i.e., no discretization), the b-value can be estimated by the maximum likelihood estimation (Aki, 1965):

$$
\hat{b} = \frac{\log e}{1/n \sum_{i=1}^n m_i - m_c},  \tag{1}
$$

where $m_c$ is the magnitude above which all events can be detected (also called magnitude of completeness) and $m_i$ are the individual magnitudes within the catalog. The uncertainty of the estimate, which stems from the fact that the sample is finite, is (Shi and Bolt, 1892):

$$
\sigma(\hat{b}) = \frac{\ln(10)b^2}{\sqrt{n_m-1}}\text{var}^{1/2}(m),  \tag{2}
$$

where $\text{var }(m)$ is the variance of the magnitudes which can be estimated from the data.

Below we discuss shortly the effect of discretized magnitufdes and two alternative methods of b-value estimation that reduce the effect of short term aftershock incompleteness.

### 1.1 Binning of magnitudes
In most cases, the magitudes are discretized. Utsu (1966) came up with an approximative equation for this case, effectively by shifting the magnitudes by half a bin. However, the correct (unbiased) estimate takes a slightly more complicated form (see Tinti and Mulargia, 1987), which we implemented for all available b-value estimators of the package (besides the one that uses Utsu's formula). For all esimtators, $\Delta m$ is a mandatory input. Of course, it can be also set to zero if no discretization is present. For most catalogs of natural seismicity, however, $\Delta m$ is between $0.2$ and $0.01$. We did not adjust the uncertainty estimate (Eq. 2) for the binning, but we did carry out synthetic tests to assure ourselves that the differences are insignificant.

### 1.2 b-positive
In an effort to reduce the impact of short term incompleteness, Van der Elst (2021) introduced the b-positive estimate. This method effectively assumes that the completeness at each point of time is determined by the last earthquake magnitude measured plus some extra quantity ($\text{d}m_c$).

The estimate can be performed as follows. First, the difference to the previous magnitudes ($m_i - m_{i-1}$) are estimated. Then, only the differences that are positive and larger than a threshold, $\text{d}m_c$, are selected (hence positive). With the obtained magnitude differences, the b-value can be estimated with the maximum likelihood method. This relies on the fact that the differences of an exponentially distributed variable are Laplace distributed (also known as double exponentially).

Although the b-positive method does effectively eliminate the effect of short term aftershock incompleteness, it is still affected by normal incompleteness resulting from the capabilities of the seismic network \textcolor{red}{citation}. This effect can be reduced either by only considering the complete catalog (above $m_c$) before estimating the differences or by increasing the threshold $\text{d}m_c$. In the SeismoStats package, both the completeness ($m_c$) and threshold ($\delta m_c$) can be given as an input, with the default being $\delta m_c = \Delta m$.

### 1.2 b-more-positive.
Based on the b-positive method, Lippiello and Petrillo (2024) developed the b-more-positive method, which makes use of the same principles.  For this method, the differences are constructed in an alternative way. For each earthquake, the next earthquake with a magnitude that is larger by $\text{d}m_c$ is taken to estimate the differences. Therefore, the number of differences will be larger than in the b-positive method, where the amount of data is typically halved. However, the variance of this estimate does not become smaller as a result. As we use the uncertainty estimator of Eq. 2, which is based on the number of mangitudes used, we underestimate the uncertainty, typically by a factor of 2.

## 2. Estimation of the b-value
In SeismoStats, we provide several ways to estimate the b-value:

- Use the {ref}`BValueEstimator <reference/analysis/bvalues:Estimators>` class
- Use the function {func}`estimate_b <seismostats.analysis.estimate_b>` (this is the easiest way, [jump there](estimate-b))
- Use the method {func}`estimate_b <seismostats.Catalog.estimate_b>` native to the Catalog class (most practical if the catalog format is used [jump there](cat-estimate-b))

Below, we show examples for each method.

### 2.1 BValueEstimator
The basis of all b-value estimations in SeismoStats is the `BValueEstimator`. The BValueEstimator class defines how b-value estimation works in general: the input is at least the magnitudes, the magnitude of completeness, and the magnitude discretization. This class is then used to implement a specific method of b-value estimation. The three methods implemented for now are described above and are called {class}`ClassicBValueEstimator <seismostats.analysis.ClassicBValueEstimator>`, {class}`BPositiveBValueEstimator <seismostats.analysis.BPositiveBValueEstimator>`, {class}`BMorePositiveBValueEstimator <seismostats.analysis.BMorePositiveBValueEstimator>`, and {class}`UtsuBValueEstimator <seismostats.analysis.UtsuBValueEstimator>`.

The class can be used as follows:

```python
>>> from seismostats.analysis import ClassicBValueEstimator
>>> estimator = ClassicBValueEstimator()
>>> estimator.calculate(mags, mc, delta_m)
1.05
>>> estimator.a_value
1.05
```

Note that the estimator automatically cuts off magnitudes below $m_c$ and does not count them. This is true for all b-value estimations. Therefore, it is of crucial importance to provide the correct $m_c$. Another crucial input is the binnning $\Delta m$, the binning of the magnitudes. For the positive methods, the threshold $\delta m_c$ and the times can be given additionally. The default value of $\delta m_c$ is $\Delta m$. If the times are not given, the estimator assumes that the magnitudes are ordered in time. The estimated b-value is finally stored within the instance of the class, which we called `estimator` in our example. 

```python
>>> from seismostats.analysis import BPositiveBValueEstimator
>>> estimator = BPositiveBValueEstimator()
>>> estimator.calculate(mags, mc, delta_m, times=times, dmc=dmc)
0.98
>>> estimator.a_value
0.98
```

Note that for `BPositiveBValueEstimator` and `BMorePositiveBValueEstimator`, the parameter `mc` still is used as in the classical case: magnitudes below will be disregarded.

### 2.2 estimate_b
In order to estimate the b-value with Eq. (1), one needs only to know the magnitude of completeness and the discretization of the magnitudes, $\Delta m$.

```python
>>> from seismostats.analysis import estimate_b
>>> magnitudes = [0, 0, 1, 1, 1, 2, 3, 2, 3, 5, 6, 7]
>>> estimate_b(magnitudes, mc=1, delta_m=1)
0.169
```

Note that the function `estimate_b` automatically cuts off magnitudes below $m_c$ and does not count them. Therefore, it is of crucial importance to provide the correct $m_c$. The default method for the b-value estimation is the classical method ({class}`ClassicBValueEstimator <seismostats.analysis.ClassicBValueEstimator>`). However, it is also possible to specify which method should be used. This can be done as follows:

```python
>>> from seismostats.analysis import estimate_b, BPositiveBValueEstimator
>>> times = numpy.arange(10)
>>> estimate_b(magnitudes, mc=1, delta_m=1, method=BPositiveBValueEstimator)
0.845
```

### 2.3 cat.estimate_b()
When you have already transformed your data into a Catalog object, you can directly use the internal method of the Catalog class, which works exactly in the same way as the function shown above.

```python
>>> cat.estimate_b(mc=1, delta_m=0.1)
0.882
```

Note that, if $\Delta m$ and $m_c$ are already defined in the catalog, the method will use these values to estimate the b-value:

```python
>>> cat.mc = 1
>>> cat.delta_m = 0.1
>>> cat.estimate_a()
0.882
```

This is especially practical since these attributes are set by the the binning method and the estimate_mc methods. 
```python
>>> # First, estimate mc
>>>cat.estimate_mc_max()
>>> # Now, it is set as an attibute 
>>> cat.mc
1.0
>>> # Second, bin the magnitudes
>>> cat.bin_magnitudes(delta_m=0.1, inplace=True)
>>> cat.delta_m
0.1
>>> cat.estimate_b()
0.882
```

## References
- Aki, Keiiti. "Maximum likelihood estimate of b in the formula log N= a-bM and its confidence limits." Bull. Earthquake Res. Inst., Tokyo Univ. 43 (1965): 237-239.
- Utsu, Tokuji. "A statistical significance test of the difference in b-value between two earthquake groups." Journal of Physics of the Earth 14.2 (1966): 37-40.
- Tinti, Stefano, and Francesco Mulargia. "Confidence intervals of b values for grouped magnitudes." Bulletin of the Seismological Society of America 77.6 (1987): 2125-2134.
- Shi, Yaolin, and Bruce A. Bolt. "The standard error of the magnitude-frequency b value." Bulletin of the Seismological Society of America 72.5 (1982): 1677-1687.
- van der Elst, Nicholas J. "B‐positive: A robust estimator of aftershock magnitude distribution in transiently incomplete catalogs." Journal of Geophysical Research: Solid Earth 126.2 (2021): e2020JB021027.
- Lippiello, E., and G. Petrillo. "b‐more‐incomplete and b‐more‐positive: Insights on a robust estimator of magnitude distribution." Journal of Geophysical Research: Solid Earth 129.2 (2024): e2023JB027849.