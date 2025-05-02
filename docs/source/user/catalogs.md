# Catalogs

>[!IMPORTANT]  
>This file is still under construction. The catalog method has some functionalities that are not yet described fully. This will be added soon

The catalog class is a pandas dataframe with some extra functionalities that are usefull if you are working with earthquake catalog data.
The easiest way to create one is directly from a dataframe:

```python
>>> from seismostats import Catalog
    cat = Catalog(df)
```

Apart from all the classical dataframe operations, the object `cat` has now some extra functionalities.
You can:
1. Plot the seismicity in time, space and its magnitude distribution
2. Estimate the completeness magnitude
3. Estimate the b-value
4. Estimate the a-value
5. transform the coordinate system to a local cartesian system
6. filter the catalog for data within a polygon

## Structure
The Catalog object has to have at least a magnitude collumn. Apart from that, it can have as many collumns as the user would like. Certain methods require however other collumns, e.g.,
- time
- Latitude, Longitude 

Further, the catalog object can have attributes, `delta_m` and `mc`. These can either be set manually:
```python
>>> cat.delta_m = 0.1
>>> cat.mc = 1
```

Or, one can set them by using a method:
```python
>>> cat.bin_magnitude(delta_m=0.1)
>>> cat.delta_m
0.1
```

These attributes then will be used for the methods. For example, the b-value estimation normally needs delta_m and mc as an input. However, if this is already set in the catalog, we don't have to specify again.

Or, one can set them by using a method:
```python
>>> estimator = cat.estimate_b(delta_m = 0.1)
ValueError: Completeness magnitude (mc) needs to be set.
>>> cat.mc = 1
>>> estimator = cat.estimate_b(delta_m = 0.1)
>>> cat.b_value
0.98
```