# Catalog Formats
Basic formats to store earthquake event data. All Structures are subclassing `pandas.DataFrame`, with additional methods to estimate parameters and convert to other formats.


```{eval-rst}
.. currentmodule:: seismostats
```

## Catalog
Simple representation of an earthquake catalog, storing each single event as a row.

(constructor-cat)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: api/

    Catalog
    Catalog.from_quakeml
    Catalog.from_dict
```

### Modify Catalog

```{eval-rst}
.. autosummary::
    :toctree: api/

    Catalog.bin_magnitudes
    Catalog.strip
    Catalog.drop_ids
    Catalog.drop_uncertainties
```

### Estimate from Catalog

```{eval-rst}
.. autosummary::
    :toctree: api/

    Catalog.estimate_b
    Catalog.estimate_mc
```

(convert-cat)=
### Convert to other format

```{eval-rst}
.. autosummary::
    :toctree: api/

    Catalog.to_quakeml
```

## ForecastCatalog
Subclass of {ref}`/reference/catalog.md#catalog`, storing events from `n_catalogs` number of forecasted catalogs, distinguished by the `catalog_id` column. Holds additional methods to work with this data structure.

(constructor-fc)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: api/

    ForecastCatalog
```

(convert-fc)=
### Convert to Other Format
    
```{eval-rst}
.. autosummary::
    :toctree: api/

    ForecastCatalog.to_quakeml
```

## GRRateGrid
Subclass of `pandas.DataFrame`, storing the Gutenberg-Richter rate values for a spatial grid. Holds additional methods to work with this data structure.

(constructor-grrg)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: api/

    GRRateGrid
```

### Modify GRRateGrid

```{eval-rst}
.. autosummary::
    :toctree: api/

    GRRateGrid.strip
    GRRateGrid.add_time_index
    GRRateGrid.reindex_cell_id
```

## ForecastGRRateGrid
Subclass of {ref}`/reference/catalog.md#grrategrid`, storing the Gutenberg-Richter rate values for a `n_grids` number of forecasted grids, distinguished by the `grid_id` column. Holds additional methods to work with this data structure.

(constructor-fgrrg)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: api/

    ForecastGRRateGrid
```