# Forecast Catalog

```{eval-rst}
.. currentmodule:: seismostats
```

## ForecastCatalog
Subclass of {ref}`/reference/formats/catalog.md#catalog`, storing events from `n_catalogs` number of forecasted catalogs, distinguished by the `catalog_id` column. Holds additional methods to work with this data structure.

(constructor-fc)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    ForecastCatalog
```

(convert-fc)=
### Convert to Other Format
    
```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    ForecastCatalog.to_quakeml
```