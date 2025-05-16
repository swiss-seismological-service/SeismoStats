# Data Formats

It is possible to represent seismic event data in different ways. Most of the functions and classes in SeismoStats work with a simple "catalog" format, which lists the events in a table-like structure. For a full description of the `Catalog` class implementation and its methods, see the [Catalog](catalogs.md) documentation.  

Another popular format is [QuakeML](https://quake.ethz.ch/quakeml/QuakeML) (xml), which we recommend using for data exchange of event data including important metadata. QuakeML can be easily converted to and form the SeismoStats `Catalog` format.


## Seismostats Implementations
Additionally to the `Catalog` format, other representations of seismic event data are `GRRateGrids`, which define seismicity using GR parameters for each grid cell separately, or `GriddedMagnitudeBins`, which define the rate of magnitudes in each magnitude bin for each grid cell.

Following the convention of QuakeML-`RealValues`, numerical values can have additional columns with a suffix indicating an uncertainty attribute, for example `latitude_uncertainty` or `longitude_loweruncertainty`. The suffix `_value` indicating the reference/mean value is dropped (`latitude` instead of `latitude_value`) for simplicity and will automatically be added when converting to QuakeML.

### Catalog
Tabular representation of seismic events. At least a 'magnitude' column is required. Additional important columns are `time`, `latitude`, `longitude` and `depth`. For conversion into other formats some additional columns can be required like `magnitude_type` or `event_type`. The Catalog format is used in most of the SeismoStats functions and classes.  
Additional columns can be added as required.

| longitude | latitude | depth | time | magnitude | magnitude_type | event_type |
| --------- | -------- | ----- | ---- | --------- | -------------- | ---------- |

Additional magnitudes (of different types) can be added as columns following the naming convention `latitude_<type>`. 

For a full definition of the catalog, the following attributes can be defined on the `Catalog` class:

| Attribute        | Description                              |
| ---------------- | ---------------------------------------- |
| starttime        | From when on the catalog is valid.       |
| endtime          | Until when the catalog is valid.         |
| bounding_polygon | The area for which the catalog is valid. |
| depth_min        | Minimum depth of the catalog.            |
| depth_max        | Maximum depth of the catalog.            |
| a_value          | a-value of the catalog.                  |
| b_value          | b-value of the catalog.                  |
| mc               | mc of the catalog.                       |
| name             | Name of the catalog.                     |

#### ForecastCatalog
An extension of the `Catalog` format, which allows to define *realizations* of the same catalog. This is useful for example when generating synthetic catalogs, where each catalog is a different random realization.

The `ForecastCatalog` has an additional column `catalog_id`, which is an unique identifier for each realization. Also the attribute `n_catalogs` is available on the `ForecastCatalog` class, which defines the total number of realizations of the catalog.

### GRRateGrid
Defines seismicity using GR parameters for each grid cell separately.

| longitude_min | longitude_max | latitude_min | latitude_max | depth_min | depth_max | rate | a   | b   | alpha | mc  | m_max |
| ------------- | ------------- | ------------ | ------------ | --------- | --------- | ---- | --- | --- | ----- | --- | ----- |

For a full definition of the `GRRateGrid`, the following attributes can be defined on the `GRRateGrid` class:

| Attribute | Description                        |
| --------- | ---------------------------------- |
| starttime | From when on the catalog is valid. |
| endtime   | Until when the catalog is valid.   |
| name      | Name of the catalog.               |

#### ForecastGRRateGrid
An extension of the `GRRateGrid` format, which allows to define *realizations* of the same grid. This is useful for example when generating synthetic data, where each realization is a different random realization of the same grid.

The `ForecastGRRategrid` has an additional column `grid_id`, which is an unique identifier for each realization. Also the attribute `n_grids` is available on the `ForecastGRRategrid` class, which defines the number of realizations of the grid.

### GriddedMagnitudeBins

This format has yet to be implemented in SeismoStats.

Defines the rate of magnitudes in each magnitude bin for each grid cell. This way it is possible to define a distribution which does not strictly follow the GR law.

| longitude_min | longitude_max | latitude_min | latitude_max | depth_min | depth_max | magnitude_min | magnitude_max | rate |
| ------------- | ------------- | ------------ | ------------ | --------- | --------- | ------------- | ------------- | ---- |

For a full definition of the `GriddedMagnitudeBins`, the following attributes can be defined on the `GriddedMagnitudeBins` class:

| Attribute | Description                        |
| --------- | ---------------------------------- |
| starttime | From when on the catalog is valid. |
| endtime   | Until when the catalog is valid.   |
| name      | Name of the catalog.               |
