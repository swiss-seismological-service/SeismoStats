# 10 Minutes to SeismoStats 

Welcome to a quick overview of **SeismoStats** — a Python package which is designed to simplify the analysis of seismic catalogs and to provide a foundation for more advanced seismicity studies.

This overview introduces the core features of SeismoStats, with a focus on three primary analysis goals:

- **b-value** calculation  
- **a-value** estimation  
- **Magnitude of completeness** estimations  

Additionally, we demonstrate how to easily visualize your data and explore key features of your catalog.

---

A central feature of **SeismoStats** is the {ref}`Catalog <reference/formats/catalog:Catalog>` object, which offers a quick and flexible way to get started. It is built on top of a `pandas.DataFrame`, meaning all standard pandas methods are available and fully supported. For more information on the pandas data struture and built-in methods refer to the [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html 'pandas User Guide'). 

The Catalog class allows for easy storage, organization, and analysis of event data (e.g., magnitudes, event times). It includes:

- *Attributes*: These include the catalog data (such as  magnitudes and event times), as well as additional parameters used in the analysis. These parameters can either be user-defined or estimated using SeismoStats' built-in methods.

- *Methods*: Tools for visualization, statistical analysis, and data conversion.
Importantly, many methods both **use** and **update** the catalog's properties and data. This architecture is illustrated below:

<figure>
  <img src="../_static/catalog_class.png" alt="Alt text" width="500"/>
  <figcaption>Figure 1: Overview on catalog class.</figcaption>
</figure>

All methods available through the `Catalog` object can also be used independently, outside the catalog structure. In this case, you can pass a `numpy` array of magnitudes directly to the respective functions for analysis.
For a brief example, see the {ref}`section below</user/10minute_intro.md#Estimating-the-b-value>` or refer to the detailed guides on estimating the {ref}`b-value</user/estimate_b.md>`, {ref}`a-value</user/estimate_a.md>` or {ref}`magnitude of completeness</user/estimate_mc.md>`. 
## 1 Creating a Catalog

Catalogs can be created in multiple ways:

- `pandas.DataFrame`  
- Python dictionaries
- Existing catalogs in **QuakeML**, **OpenQuake** or **CSV** formats

and each possibility is described in the {ref}`Catalog guide </user/catalogs.md>`.

You can also fetch earthquake data directly from FDSN servers such as **EIDA** and **USGS** using the built-in FDSN client, as shown in the example below.


> **Note:** Your catalog must include a `magnitude` column. For full functionality (especially plotting and analysis), it is recommended to also include `time`, `latitude`, `longitude`, and `depth`.


### 1.1 Example: Downloading a catalog from a FDSN-Server
Since downloads are limited to a maximum of 20,000 events, you can use the batch download option for larger datasets. This method downloads the data in subsets of events based on the specified `batch_size`.
```python
>>> from seismostats.catalogs.client import FDSNWSEventClient
>>> from seismostats import Catalog
>>> import pandas as pd

>>> # Define time range and region of interest
>>> start_time = pd.to_datetime('2020/01/01')
>>> end_time = pd.to_datetime('2022/01/01')

>>> min_longitude = 5
>>> max_longitude = 11
>>> min_latitude = 45
>>> max_latitude = 48

>>> min_magnitude = 0.5
>>> url = 'http://eida.ethz.ch/fdsnws/event/1/query'
>>> client = FDSNWSEventClient(url)

>>> # Download events
>>> df = client.get_events(
...     start_time=start_time,
...     end_time=end_time,
...     min_magnitude=min_magnitude,
...     min_longitude=min_longitude,
...     max_longitude=max_longitude,
...     min_latitude=min_latitude,
...     max_latitude=max_latitude,
...     batch_size=1000)

>>> # Create catalog and preview entries
>>> cat.head()
  event_type             time                  latitude  longitude      depth evaluationmode  magnitude magnitude_type  magnitude_MLhc  magnitude_MLh
0  earthquake  2021-12-30 07:43:14.681975   46.051445   7.388025  1181.640625        manual   2.510115           MLhc     2.510115344            NaN
1  earthquake  2021-12-30 01:35:37.014056   46.778985   9.476219  9294.921875        manual   1.352086           MLhc     1.352086067            NaN
2  earthquake  2021-12-29 08:48:59.059653   47.779511   7.722354 16307.812500        manual   0.817480           MLhc     0.8174796651           NaN

```

---

## 2 Visualizing the Catalog

The `Catalog` class offers several built-in methods for analyzing and visualizing your seismic data.

These tools allow you to:

- Quickly explore spatial and temporal patterns
- Inspect magnitude distributions
- Generate publication-ready plots with minimal code

Use these methods to gain insights into your catalog before performing more advanced statistical analyses.

```python
>>> # plot the location of the events on a map
>>> cat.plot_in_space(include_map=True)

>>> # plot all available magnitudes over time:
>>> cat.plot_mags_in_time()

>>> # plot the cumulative frequency-magnitude distribution with bin size 0.1
>>> cat.plot_cum_fmd(delta_m=0.1)
```
<figure>
  <img src="../_static/catalog_map.png" alt="Alt text" width="500"/>
  <figcaption>Figure 2: Map with Earthquake locations.</figcaption>
</figure>

<figure>
  <img src="../_static/mag_vs_time.png" alt="Alt text" width="500"/>
  <figcaption>Figure 3: Event magnitudes over time.</figcaption>
</figure>

<figure>
  <img src="../_static/cum_fmd_single.png" alt="Alt text" width="500"/>
  <figcaption>Figure 4: Cumulative frequency-magnitude distribution.</figcaption>
</figure>


All available plotting methods are described in more detail in the {ref}`Plotting guide </user/plots.md>`.

---

## 3 First analysis 
Before performing statistical analysis, it's important to bin the magnitudes in your catalog correctly.  
The choice of bin size should reflect the magnitude resolution of your dataset to ensure accurate results.

Proper binning is essential for calculating meaningful b-values, a-values, and the magnitude of completeness.

By using `inplace=True` in the `bin_magnitudes` method, the magnitudes of the catalog object will be replaced by their binned version:

```python
>>> # The magnitudes of the first events of the original catalog
>>> print(cat.magnitude.head())
0    2.510115
1    1.352086
2    0.817480
3    1.252432
4    0.897306
Name: magnitude, dtype: float64

>>> # Now we set delta_m and bin the magnitudes accordingly
>>> cat.delta_m = 0.1
>>> cat.bin_magnitudes(inplace=True)

>>> # Using inplace=True, the magnitudes of the catalog are overwritten by 
>>> # the binned version:
>>> print(cat.magnitude.head())
0    2.5
1    1.4
2    0.8
3    1.3
4    0.9
Name: magnitude, dtype: float64
```

### 3.1 Estimating the Magnitude of Completeness

Seismostats provides three methods to estimate the **magnitude of completeness** ($M_c$) in earthquake catalogs:

- **Maximum Curvature**
- **b-Stability**
- **Kolmogorov-Smirnov (KS) Test**

These methods help assess the quality of your catalog by identifying the lowest magnitude above which events are reliably recorded. More information on the methods can be found in the section {doc}`Magnitude of Completeness<estimate_mc>`.

> **Note:** 
> Calling any of the methods below will overwrite the `Catalog.mc` property with the newly estimated magnitude of completeness.

```python
>>> cat.estimate_mc_maxc(fmd_bin=0.1)
>>> print(cat.mc)
1.0
>>> cat.estimate_mc_b_stability()
>>> print(cat.mc)
1.5
>>> cat.estimate_mc_ks()
>>> print(cat.mc)
2.1
```

### 3.2 Estimating the b-value
The **b-value** in the Gutenberg-Richter law quantifies the relative frequency of large versus small earthquakes in a seismic catalog. 
The most common approach to estimate the b-value is through the **maximum likelihood method**, assuming an exponential distribution of magnitudes. Additional estimation techniques are discussed in the section on {doc}`b-value estimations <estimate_b>`.

Before estimating the b-value, make sure that the properties `Catalog.mc` and `Catalog.delta_m` are set. Alternatively, these parameter can be directly provided when calling `estimate_b`.

You can also estimate the b-value independently of the Catalog object by passing a numpy array of magnitudes to {func}`estimate_b <seismostats.analysis.estimate_b>`.
```python
>>> # Estimate b with the catalog method and the internal attributes
>>> cat.mc = 1.8
>>> cat.delta_m = 0.1
>>> cat.estimate_b()
>>> print(cat.b_value)
1.064816286818266

>>> # Estimate b with the catalog method and additional arguments
>>> cat.estimate_b(delta_m = 0.1, mc=1.8)
>>> print(cat.b_value)
1.064816286818266

>>> # Estimate b independently of the catalog class
>>> from seismostats.analysis import estimate_b
>>> b_value = estimate_b(magnitudes = cat.magnitude, delta_m = 0.1, mc=1.8)
>>> print(b_value)
1.064816286818266
```

### 3.3 Estimating the a-value
The **a-value** of the Gutenberg-Richter law  describes the overall earthquake activity in a specific area and time span. It reflects how many events are expected, regardless of their magnitude. Further discussions on the a-value can be found in the section {doc}`a-value estimations <estimate_a>`.

Similar to the b-value estimations, the parameter `Catalog.mc`, `Catalog.delta_m` must be defined beforehand or provided directly as arguments to the a-value estimation method. 
```python
>>> # Estimate a with the catalog method and the internal attributes
>>> cat.estimate_a()
>>> print(cat.a_value)
2.2121876044039577

>>> # Estimate a with the catalog method and additional arguments
>>> cat.estimate_a(delta_m = 0.1, mc=1.8)
>>> print(cat.a_value)
2.2121876044039577
```

