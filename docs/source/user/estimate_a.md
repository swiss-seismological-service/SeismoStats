
# Estimate a-value
The a-value is the parameter in the Gutenberg-Richter law that contains information on the rate of seismicity within the volume and time interval of interest. Here, we will briefly explain how to estimate the a-value, how to scale it correctly so that it can be compared across catalogs, and explain the different methods available.

## 1. Short Introduction on a-value Estimation
The GR law above the completeness magnitude $m_c$ can be expressed as follows:

$$
N(m) = 10^{a - b (m - m_c)},
$$

where $N(m)$ is the number of events with magnitudes larger than or equal to $m$ that occurred in the catalog. With this definition of the a-value, we can estimate the a-value as the logarithm of the number of earthquakes above completeness:

$$
a = \log N(m_c).  	ag{1}
$$

There are, however, two commonly used modifications of the function above, which are relevant if we want to compare different a-values to each other.

### 1.1 Reference Magnitude
First, it might be that the level of completeness is not constant. Therefore, in practice, many instead estimate the a-value with respect to a certain reference magnitude, such that $10^{a_{m_{ref}}} = N(m_{ref})$. Here, $N(m_{ref})$ is not the actual number of earthquakes above $m_{ref}$, but the extrapolated number if the GR law were perfectly valid above and below $m_c$, as shown in Fig. 1 below. The new a-value can now be estimated using the a-value defined in eq. (1): $a_{m_{ref}} = a - b(m_{ref} - m_c)$.

<figure>
  <img src="../_static/a_value_reference.png" alt="Alt text" width="400"/>
  <figcaption>Figure 1: Sketch of the reference magnitude.</figcaption>
</figure>

### 1.2 Scaling
Second, the time intervals that are compared are often not the same, making it hard to compare. In many cases, researchers scale the a-value so that $N(m)$ means the number of earthquakes above $m$ within a year (which is effectively a rate). We include this possibility in the form of a scaling factor. This scaling factor encompasses information about how many time-units fit within the time interval of observation. E.g., if the interval of observation is 10 years but we want to scale the a-value to one year, the scaling factor is 10. Note that the same can be applied for spatial comparison: If we want to compare the number of earthquakes in two different volumes, we might be interested in the number of earthquakes per cubic km. If we have a volume of 100 cubic km, the scaling factor is therefore 100.

### 1.3 Positive Methods
Finally, similarly to the b-value estimation, it was proposed to estimate the a-value by taking only the earthquakes that are larger than the previous one into account. This is built on the implicit assumption that the momentary completeness is given by the magnitude of the last detected earthquake (plus a buffer, called $\delta m_c$). We based our definitions on the article by Van der Elst and Page, 2023, with the difference that we homogenized the naming convention with the b-value methods:

- "positive" means that for ( )
- "more positive" means that (the method is taken from Van der Elst and Page, 2023), but the naming is taken

## 2. Estimation of the a-value
In SeismoStats, we provide several ways to estimate the a-value:

- Use the {class}`AValueEstimator <seismostats.analysis.AValueEstimator>` class
- Use the function {func}`estimate_a <seismostats.analysis.estimate_a>` (this is the easiest way, [jump there](estimate-a))
- Use the Catalog class `estimate_a` method (most practical if the catalog format is used [jump there](cat-estimate-a))

Below, we explain each method.

### 2.1 AValueEstimator
The basis of all a-value estimations in SeismoStats is the `AValueEstimator`. The AValueEstimator class defines how a-value estimation works in general: the input is at least the magnitudes, the magnitude of completeness, and the magnitude discretization. This class is then used to implement a specific method of a-value estimation. The three methods implemented for now are described above and are called `ClassicAValueEstimator`, `APositiveAValueEstimator`, and `AMorePositiveAValueEstimator`. These classes function in a very similar way to the b-value estimators described in {doc}`estimate b <estimate_b>`.

The class can be used as follows:

```python
>>> from seismostats.analysis import ClassicAValueEstimator
>>> estimator = ClassicAValueEstimator()
>>> estimator.calculate(mags, mc, delta_m)
np.float64(3)
>>> estimator.a_value
np.float64(3)
```

In the example above, `mags` is a vector of magnitudes with 1000 values above $m_c$. Note that the estimator automatically cuts off magnitudes below $m_c$ and does not count them. This is true for all a-value estimations. Therefore, it is of crucial importance to provide the correct $m_c$. The reason that $\Delta m$ is needed here is only to correctly cut off at $m_c$. The estimated a-value is finally stored within the instance of the class, which we called `estimator` in our example.

`APositiveAValueEstimator` and `AMorePositiveAValueEstimator` work in a similar way. However, they have the additional possible arguments `dmc` (see $\delta m_c$ above) and `time`. If `dmc` is not given, it is set to $\Delta m$. If `time` is given, the estimator will assume that the magnitudes are already ordered in time.

```python
>>> from seismostats.analysis import AMorePositiveAValueEstimator
>>> estimator = ClassicAValueEstimator()
>>> estimator.calculate(mags, mc, delta_m, dmc=dmc, time=time)
np.float64(1.5)
>>> estimator.a_value
np.float64(1.5)
```

Note that for `APositiveAValueEstimator` and `AMorePositiveAValueEstimator`, the parameter `mc` still cuts the original magnitudes.

### 2.2 estimate_a
In order to estimate the a-value with eq. (1), one needs only to know the magnitude of completeness and the discretization of the magnitudes, $\Delta m$.

```python
>>> from seismostats.analysis import estimate_a
>>> magnitudes = [0, 0, 1, 1, 1, 2, 3, 2, 3, 5, 6, 7]
>>> estimate_a(magnitudes, mc=1, delta_m=1)
np.float64(1)
```

Note that the function `estimate_a` automatically cuts off magnitudes below $m_c$ and does not count them. This is true for all a-value functionalities. Therefore, it is of crucial importance to provide the correct $m_c$. The reason that $\Delta m$ is needed here is only to correctly cut off at $m_c$.

### 2.3 cat.estimate_a
When you have already transformed your data into a Catalog object, you can directly use the internal method of the Catalog class, which works exactly in the same way as the function shown above.

```python
>>> cat.estimate_a(mc=1, delta_m=1)
np.float64(1)
```

Note that, if $\Delta m$ and $m_c$ are already defined in the catalog, the method will use these values to estimate the a-value:

```python
>>> cat.mc = 1
>>> cat.delta_m = 1
>>> cat.estimate_a()
np.float64(1)
```

This is practical: if 

### References
- Van der Elst, Nicholas J., and Morgan T. Page. "a‐positive: A robust estimator of the earthquake rate in incomplete or saturated catalogs." *Journal of Geophysical Research: Solid Earth* 128.10 (2023): e2023JB027089.
