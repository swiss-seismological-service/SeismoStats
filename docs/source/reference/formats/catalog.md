# Catalog


```{eval-rst}
.. currentmodule:: seismostats
```

## Catalog
Simple representation of an earthquake catalog, storing each single event as a row.

(constructor-cat)=
### Constructor

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :template: ../_templates/autosummary/class_notoctree.rst
    :nosignatures:

    Catalog
```
```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.from_quakeml
    Catalog.from_dict
```

### Modify Catalog

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.bin_magnitudes
    Catalog.strip
    Catalog.drop_ids
    Catalog.drop_uncertainties
```

### Estimate from Catalog

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.estimate_b
    Catalog.estimate_mc
```

(convert-cat)=
### Convert to other format

```{eval-rst}
.. autosummary::
    :toctree: ../api/
    :nosignatures:

    Catalog.to_quakeml
```