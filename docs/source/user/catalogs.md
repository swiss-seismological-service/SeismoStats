# Catalogs

The catalog class is a pandas dataframe with some extra functionalities that are usefull if you are working with earthquake catalog data.
The easiest way to create one is directly from a dataframe:

```python
>>> from seismostats import Catalog
    cat = Catalog(df)
```

Apart from all the classical dataframe operations, the object `cat` has now some extra functionalities.
You can:
1. Plot the seismicity in time, space and its magnitude distribution
1. Estimate the completeness magnitude
2. Estimate the b-value
3. Estimate the a-value
4. transform the coordinate system to a local cartesian system
5. filter the catalog for data within a polygon

![catalog_plot](../_static/catalog_plots_mc_None.png "Overview on catalog properties")